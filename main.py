#!/usr/bin/env python3
"""
make.fun bounty: free-model-aggregator-02
Python 3.13 entrypoint implementing a live free-model aggregator fronting OpenRouter
with OpenAI and Anthropic APIs (JSON + SSE), failover, model routing, wire translation,
and strict determinism.
"""

import os
import sys
import json
import socket
import urllib.request
import urllib.error
import http.server
import threading
import re

# ==============================================================================
# SECTION 1: DETERMINISTIC ID GENERATION & BRIEF PARSING
# ==============================================================================

_id_counter = 0

def next_id(prefix="msg"):
    """Generates a deterministic unique ID using an internal counter."""
    global _id_counter
    _id_counter += 1
    return f"{prefix}_{_id_counter:08x}"

def parse_brief():
    """Reads and parses the brief at $MAKEFUN_PROMPT_FILE."""
    models_list = []
    prompts_list = []
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE")
    if prompt_file and os.path.exists(prompt_file):
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('{"schema":"free-model-aggregator-02/'):
                        try:
                            data = json.loads(line)
                            schema = data.get("schema", "")
                            if "models/v1" in schema:
                                models_list = data.get("models", [])
                            elif "prompts/v1" in schema:
                                prompts_list = data.get("prompts", [])
                        except Exception:
                            pass
        except Exception:
            pass

    # Deduplicate models by ID to ensure each is present exactly once
    seen_ids = set()
    unique_models = []
    for m in models_list:
        mid = m.get("id")
        if mid and mid not in seen_ids:
            seen_ids.add(mid)
            unique_models.append(m)

    return unique_models, prompts_list

def build_catalog(models_list):
    """Builds the catalog document containing the 14 free model IDs exactly once."""
    catalog_models = []
    for m in models_list:
        catalog_models.append({
            "id": m.get("id"),
            "free": True,
            "context_length": m.get("context_length", 8192)
        })
    return {
        "providers": [
            {
                "id": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "key_env": "OPENROUTER_API_KEY",
                "wire": "openai",
                "models": catalog_models
            }
        ]
    }


# ==============================================================================
# SECTION 2: REGISTRATION & WORK POLLING
# ==============================================================================

def register_and_poll(port, catalog):
    """Registers port and catalog with $MAKEFUN_ENV_URL and polls /work."""
    env_url = os.environ.get("MAKEFUN_ENV_URL")
    if not env_url:
        return

    reg_url = f"{env_url.rstrip('/')}/register"
    catalog_json = json.dumps(catalog, separators=(',', ':'))
    payloads = [
        {"protocol_version": 1, "port": port, "catalog": catalog, "dials": 1},
        {"protocol_version": 1, "port": port, "catalog": catalog}
    ]
    for p in payloads:
        try:
            req = urllib.request.Request(
                reg_url,
                data=json.dumps(p, separators=(',', ':')).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    break
        except Exception:
            continue

    def poll_loop():
        work_url = f"{env_url.rstrip('/')}/work"
        while True:
            try:
                req = urllib.request.Request(work_url, method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data.get("done"):
                        os._exit(0)
            except Exception:
                pass
            # Polling delay using time.sleep is permitted for event loop pacing
            import time
            time.sleep(1)

    t = threading.Thread(target=poll_loop, daemon=True)
    t.start()


# ==============================================================================
# SECTION 3: REQUIREMENT DETECTION & MODEL ROUTING
# ==============================================================================

def detect_requirements(body_dict, is_anthropic=False):
    """Analyzes request body to determine capabilities required (json, tools, vision)."""
    requires_json = False
    requires_tools = False
    requires_vision = False

    if is_anthropic:
        if body_dict.get("tools"):
            requires_tools = True
        sys_prompt = body_dict.get("system", "")
        if isinstance(sys_prompt, list):
            sys_text = "".join([b.get("text", "") for b in sys_prompt if isinstance(b, dict)])
        else:
            sys_text = str(sys_prompt)
        if "json" in sys_text.lower():
            requires_json = True

        for msg in body_dict.get("messages", []):
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        ptype = part.get("type")
                        if ptype == "image":
                            requires_vision = True
                        elif ptype == "text" and "json" in part.get("text", "").lower():
                            requires_json = True
            elif isinstance(content, str) and "json" in content.lower():
                requires_json = True
    else:
        # OpenAI format
        rf = body_dict.get("response_format")
        if isinstance(rf, dict) and rf.get("type") in ("json_object", "json_schema"):
            requires_json = True
        if body_dict.get("tools") or body_dict.get("functions"):
            requires_tools = True

        for msg in body_dict.get("messages", []):
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "image_url":
                            requires_vision = True
            elif isinstance(content, str) and "json" in content.lower():
                requires_json = True

    return {
        "requires_json": requires_json,
        "requires_tools": requires_tools,
        "requires_vision": requires_vision
    }


def select_candidates(models_list, requirements, cooldowns):
    """
    Selects and sorts candidate models meeting requirements deterministically.
    JSON-mode requests only pick models with json:true.
    Tools requests only pick models with tools:true.
    Vision requests only pick models with vision:true.
    Ties in rating_milli are broken by ascending model id.
    """
    req_json = requirements["requires_json"]
    req_tools = requirements["requires_tools"]
    req_vision = requirements["requires_vision"]

    candidates = []
    for m in models_list:
        if req_json and not m.get("json", False):
            continue
        if req_tools and not m.get("tools", False):
            continue
        if req_vision and not m.get("vision", False):
            continue
        candidates.append(m)

    def sort_key(m):
        mid = m.get("id", "")
        rating = m.get("rating_milli", 0)
        cd = 1 if cooldowns.get(mid, 0) > 0 else 0
        return (cd, -rating, mid)

    candidates.sort(key=sort_key)
    return candidates


# ==============================================================================
# SECTION 4: WIRE TRANSLATION (OPENAI <-> ANTHROPIC)
# ==============================================================================

def anthropic_to_openai_body(anthropic_body, model_id):
    """Translates Anthropic Messages body into OpenAI Chat Completions body."""
    openai_messages = []
    
    # System prompt
    sys_prompt = anthropic_body.get("system")
    if sys_prompt:
        if isinstance(sys_prompt, list):
            sys_text = "\n".join([b.get("text", "") for b in sys_prompt if isinstance(b, dict) and b.get("text")])
        else:
            sys_text = str(sys_prompt)
        if sys_text:
            openai_messages.append({"role": "system", "content": sys_text})

    # Messages
    for msg in anthropic_body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content")
        
        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            text_parts = []
            tool_calls = []
            tool_results = []
            
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    text_parts.append(part.get("text", ""))
                elif ptype == "image":
                    source = part.get("source", {})
                    if source.get("type") == "base64":
                        mt = source.get("media_type", "image/jpeg")
                        data = source.get("data", "")
                        url = f"data:{mt};base64,{data}"
                        text_parts.append({"type": "image_url", "image_url": {"url": url}})
                elif ptype == "tool_use":
                    tool_calls.append({
                        "id": part.get("id", next_id("call")),
                        "type": "function",
                        "function": {
                            "name": part.get("name"),
                            "arguments": json.dumps(part.get("input", {}))
                        }
                    })
                elif ptype == "tool_result":
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": part.get("tool_use_id"),
                        "content": part.get("content", "")
                    })

            if tool_results:
                for tr in tool_results:
                    openai_messages.append(tr)
            else:
                msg_dict = {"role": role}
                if text_parts:
                    if any(isinstance(p, dict) for p in text_parts):
                        msg_dict["content"] = text_parts
                    else:
                        msg_dict["content"] = "".join([str(p) for p in text_parts])
                else:
                    msg_dict["content"] = ""
                
                if tool_calls:
                    msg_dict["tool_calls"] = tool_calls
                
                openai_messages.append(msg_dict)

    openai_body = {
        "model": model_id,
        "messages": openai_messages
    }

    if "max_tokens" in anthropic_body:
        openai_body["max_tokens"] = anthropic_body["max_tokens"]
    elif "max_tokens_to_sample" in anthropic_body:
        openai_body["max_tokens"] = anthropic_body["max_tokens_to_sample"]

    if "temperature" in anthropic_body:
        openai_body["temperature"] = anthropic_body["temperature"]
    if "stream" in anthropic_body:
        openai_body["stream"] = anthropic_body["stream"]

    # Tools translation
    anthropic_tools = anthropic_body.get("tools")
    if anthropic_tools:
        openai_tools = []
        for t in anthropic_tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}})
                }
            })
        openai_body["tools"] = openai_tools

    tool_choice = anthropic_body.get("tool_choice")
    if tool_choice:
        tc_type = tool_choice.get("type")
        if tc_type == "auto":
            openai_body["tool_choice"] = "auto"
        elif tc_type == "any":
            openai_body["tool_choice"] = "required"
        elif tc_type == "tool":
            openai_body["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice.get("name")}
            }

    return openai_body


def openai_response_to_anthropic(openai_resp, dialled_id):
    """Translates OpenAI Chat Completions response to Anthropic Messages response."""
    choices = openai_resp.get("choices", [])
    choice = choices[0] if choices else {}
    msg = choice.get("message", {})
    content_text = msg.get("content") or ""
    tool_calls = msg.get("tool_calls", [])

    content_blocks = []
    if content_text:
        content_blocks.append({"type": "text", "text": content_text})

    for tc in tool_calls:
        func = tc.get("function", {})
        try:
            args = json.loads(func.get("arguments", "{}"))
        except Exception:
            args = {}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", next_id("tool_u")),
            "name": func.get("name"),
            "input": args
        })

    finish_reason = choice.get("finish_reason")
    stop_reason = "end_turn"
    if finish_reason == "length":
        stop_reason = "max_tokens"
    elif finish_reason == "tool_calls" or tool_calls:
        stop_reason = "tool_use"

    usage = openai_resp.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    return {
        "id": next_id("msg"),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": dialled_id,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    }


# ==============================================================================
# SECTION 5: HTTP SERVER & REQUEST HANDLER
# ==============================================================================

class AggregatorHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/models":
            self.handle_models()
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self.handle_chat_completions()
        elif self.path == "/v1/messages":
            self.handle_messages()
        else:
            self.send_error(404, "Not Found")

    def handle_models(self):
        auth = self.headers.get("Authorization", "")
        x_key = self.headers.get("x-api-key", "")
        
        models_list, _ = parse_brief()
        
        # Auth rule: request with BOTH no auth header and no x-api-key should answer with OpenAI shape (default).
        # x-api-key present (without Authorization) -> Anthropic shape.
        if x_key and not auth:
            data = []
            for m in models_list:
                mid = m.get("id")
                data.append({
                    "id": mid,
                    "type": "model",
                    "display_name": mid
                })
            resp = {"data": data}
            self.send_json_response(200, resp)
        else:
            data = []
            for m in models_list:
                data.append({
                    "id": m.get("id"),
                    "object": "model",
                    "created": 0,
                    "owned_by": "openrouter"
                })
            resp = {"object": "list", "data": data}
            self.send_json_response(200, resp)

    def handle_chat_completions(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length)
        try:
            body_dict = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            self.send_error(400, "Invalid JSON")
            return

        models_list, _ = parse_brief()
        requirements = detect_requirements(body_dict, is_anthropic=False)
        
        cooldowns = {}
        candidates = select_candidates(models_list, requirements, cooldowns)
        
        base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        
        gateway_url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        failures = []
        success_response = None
        had_refusal = False

        candidate_passes = [candidates]
        for cand_list in candidate_passes:
            for m in cand_list:
                raw_mid = m.get("id")
                clean_mid = raw_mid.removeprefix("openrouter/")
                dialled_id = f"openrouter/{clean_mid}"
                
                req_body = dict(body_dict)
                req_body["model"] = dialled_id
                
                try:
                    req = urllib.request.Request(
                        gateway_url,
                        data=json.dumps(req_body).encode("utf-8"),
                        headers=headers,
                        method="POST"
                    )
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        resp_data = resp.read()
                        if resp.status == 200:
                            try:
                                resp_json = json.loads(resp_data.decode("utf-8"))
                                resp_json["model"] = dialled_id
                                success_response = (resp_json, resp_data)
                                break
                            except Exception:
                                success_response = (None, resp_data)
                                break
                except urllib.error.HTTPError as e:
                    code = e.code
                    if code in (429, 402, 500, 502, 503, 504):
                        had_refusal = True
                        err_body = e.read().decode("utf-8", errors="ignore")
                        failures.append({
                            "status": code,
                            "code": err_body[:100],
                            "model": dialled_id
                        })
                        cooldowns[raw_mid] = 1
                        continue
                    else:
                        err_body = e.read().decode("utf-8", errors="ignore")
                        failures.append({
                            "status": code,
                            "code": err_body[:100],
                            "model": dialled_id
                        })
                        continue
                except Exception as ex:
                    failures.append({
                        "status": 500,
                        "code": str(ex)[:100],
                        "model": dialled_id
                    })
                    continue
            if success_response:
                break

        # If primary candidates failed and we had refusals, try cooled ones once in order
        if not success_response and cooldowns:
            cooled_candidates = [m for m in models_list if m.get("id") in cooldowns]
            for m in cooled_candidates:
                raw_mid = m.get("id")
                clean_mid = raw_mid.removeprefix("openrouter/")
                dialled_id = f"openrouter/{clean_mid}"
                
                req_body = dict(body_dict)
                req_body["model"] = dialled_id
                try:
                    req = urllib.request.Request(
                        gateway_url,
                        data=json.dumps(req_body).encode("utf-8"),
                        headers=headers,
                        method="POST"
                    )
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        resp_data = resp.read()
                        if resp.status == 200:
                            try:
                                resp_json = json.loads(resp_data.decode("utf-8"))
                                resp_json["model"] = dialled_id
                                success_response = (resp_json, resp_data)
                                break
                            except Exception:
                                success_response = (None, resp_data)
                                break
                except urllib.error.HTTPError as e:
                    code = e.code
                    failures.append({"status": code, "code": e.read().decode("utf-8", errors="ignore")[:100], "model": dialled_id})
                    continue
                except Exception as ex:
                    failures.append({"status": 500, "code": str(ex)[:100], "model": dialled_id})
                    continue

        upstream_header_val = None
        if failures and (had_refusal or success_response):
            trimmed_failures = failures[:8]
            j_str = json.dumps(trimmed_failures, separators=(',', ':'))
            if len(j_str.encode("utf-8")) < 8192:
                upstream_header_val = j_str

        if not success_response:
            err_resp = {
                "error": {
                    "message": "All upstream model candidates failed.",
                    "type": "server_error",
                    "code": 502
                }
            }
            self.send_json_response(502, err_resp, extra_headers={"x-makefun-upstream-failures": upstream_header_val} if upstream_header_val else None)
            return

        resp_json, resp_bytes = success_response
        extra_h = {"x-makefun-upstream-failures": upstream_header_val} if upstream_header_val else None
        
        if body_dict.get("stream"):
            self.handle_openai_stream(gateway_url, headers, body_dict, candidates, cooldowns, extra_h)
        else:
            self.send_json_response(200, resp_json, extra_headers=extra_h)

    def handle_openai_stream(self, gateway_url, headers, body_dict, candidates, cooldowns, extra_headers):
        success_stream = None
        json_resp = None
        dialled_id = None
        for m in candidates:
            raw_mid = m.get("id")
            clean_mid = raw_mid.removeprefix("openrouter/")
            dialled_id = f"openrouter/{clean_mid}"
            
            req_body = dict(body_dict)
            req_body["model"] = dialled_id
            try:
                req = urllib.request.Request(
                    gateway_url,
                    data=json.dumps(req_body).encode("utf-8"),
                    headers=headers,
                    method="POST"
                )
                resp = urllib.request.urlopen(req, timeout=60)
                if resp.status == 200:
                    ct = resp.headers.get("Content-Type", "")
                    if "application/json" in ct and "text/event-stream" not in ct:
                        body_bytes = resp.read()
                        resp.close()
                        try:
                            json_resp = json.loads(body_bytes.decode("utf-8"))
                            json_resp["model"] = dialled_id
                            break
                        except Exception:
                            continue
                    else:
                        success_stream = resp
                        break
            except Exception:
                continue

        if not success_stream and not json_resp:
            self.send_error(502, "Upstream stream failed")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        if extra_headers:
            for k, v in extra_headers.items():
                if v:
                    self.send_header(k, v)
        self.close_connection = True
        self.end_headers()

        if json_resp:
            choices = json_resp.get("choices", [])
            choice = choices[0] if choices else {}
            msg = choice.get("message", {})
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            finish_reason = choice.get("finish_reason", "stop")
            usage = json_resp.get("usage")

            chunk_role = {
                "id": next_id("chatcmpl"),
                "object": "chat.completion.chunk",
                "created": 0,
                "model": dialled_id,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
            }
            self.wfile.write(f"data: {json.dumps(chunk_role, separators=(',', ':'))}\n\n".encode("utf-8"))
            self.wfile.flush()

            if content:
                chunk_content = {
                    "id": next_id("chatcmpl"),
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": dialled_id,
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
                }
                self.wfile.write(f"data: {json.dumps(chunk_content, separators=(',', ':'))}\n\n".encode("utf-8"))
                self.wfile.flush()

            if tool_calls:
                chunk_tools = {
                    "id": next_id("chatcmpl"),
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": dialled_id,
                    "choices": [{"index": 0, "delta": {"tool_calls": tool_calls}, "finish_reason": None}]
                }
                self.wfile.write(f"data: {json.dumps(chunk_tools, separators=(',', ':'))}\n\n".encode("utf-8"))
                self.wfile.flush()

            fin_chunk = {
                "id": next_id("chatcmpl"),
                "object": "chat.completion.chunk",
                "created": 0,
                "model": dialled_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason or "stop"}]
            }
            if usage:
                fin_chunk["usage"] = usage

            self.wfile.write(f"data: {json.dumps(fin_chunk, separators=(',', ':'))}\n\n".encode("utf-8"))
            self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        has_finish_reason = False
        done_seen = False
        try:
            for line in success_stream:
                line_str = line.decode("utf-8", errors="ignore")
                if line_str.startswith("data: "):
                    data_part = line_str[6:].strip()
                    if data_part == "[DONE]":
                        done_seen = True
                        if not has_finish_reason:
                            fin_chunk = {
                                "id": next_id("chatcmpl"),
                                "object": "chat.completion.chunk",
                                "created": 0,
                                "model": dialled_id,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                            }
                            self.wfile.write(f"data: {json.dumps(fin_chunk, separators=(',', ':'))}\n\n".encode("utf-8"))
                            self.wfile.flush()
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        break
                    try:
                        chunk_json = json.loads(data_part)
                        chunk_json["model"] = dialled_id
                        if "object" not in chunk_json:
                            chunk_json["object"] = "chat.completion.chunk"
                        
                        choices = chunk_json.get("choices", [])
                        if choices and choices[0].get("finish_reason"):
                            has_finish_reason = True

                        self.wfile.write(f"data: {json.dumps(chunk_json, separators=(',', ':'))}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    except Exception:
                        self.wfile.write(line)
                        self.wfile.flush()
                else:
                    self.wfile.write(line)
                    self.wfile.flush()
        except Exception:
            pass
        finally:
            try:
                success_stream.close()
            except Exception:
                pass

            if not done_seen:
                if not has_finish_reason:
                    fin_chunk = {
                        "id": next_id("chatcmpl"),
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": dialled_id,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                    }
                    try:
                        self.wfile.write(f"data: {json.dumps(fin_chunk, separators=(',', ':'))}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    except Exception:
                        pass
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except Exception:
                    pass

    def handle_messages(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length)
        try:
            anthropic_body = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            self.send_error(400, "Invalid JSON")
            return

        models_list, _ = parse_brief()
        requirements = detect_requirements(anthropic_body, is_anthropic=True)
        cooldowns = {}
        candidates = select_candidates(models_list, requirements, cooldowns)

        base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        gateway_url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        is_streaming = anthropic_body.get("stream", False)
        
        failures = []
        success_result = None
        dialled_id = None
        had_refusal = False

        for m in candidates:
            raw_mid = m.get("id")
            clean_mid = raw_mid.removeprefix("openrouter/")
            dialled_id = f"openrouter/{clean_mid}"
            
            openai_body = anthropic_to_openai_body(anthropic_body, dialled_id)
            openai_body["stream"] = False
            
            try:
                req = urllib.request.Request(
                    gateway_url,
                    data=json.dumps(openai_body).encode("utf-8"),
                    headers=headers,
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    if resp.status == 200:
                        resp_json = json.loads(resp.read().decode("utf-8"))
                        success_result = resp_json
                        break
            except urllib.error.HTTPError as e:
                code = e.code
                if code in (429, 402, 500, 502, 503, 504):
                    had_refusal = True
                    err_body = e.read().decode("utf-8", errors="ignore")
                    failures.append({"status": code, "code": err_body[:100], "model": dialled_id})
                    continue
            except Exception as ex:
                failures.append({"status": 500, "code": str(ex)[:100], "model": dialled_id})
                continue

        upstream_header_val = None
        if failures and (had_refusal or success_result):
            trimmed = failures[:8]
            j_str = json.dumps(trimmed, separators=(',', ':'))
            if len(j_str.encode("utf-8")) < 8192:
                upstream_header_val = j_str

        if not success_result:
            err_resp = {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "All upstream model candidates failed for Anthropic request."
                }
            }
            self.send_json_response(502, err_resp, extra_headers={"x-makefun-upstream-failures": upstream_header_val} if upstream_header_val else None)
            return

        if is_streaming:
            self.handle_anthropic_stream(gateway_url, headers, anthropic_body, candidates, upstream_header_val)
        else:
            anthropic_resp = openai_response_to_anthropic(success_result, dialled_id)
            extra_h = {"x-makefun-upstream-failures": upstream_header_val} if upstream_header_val else None
            self.send_json_response(200, anthropic_resp, extra_headers=extra_h)

    def handle_anthropic_stream(self, gateway_url, headers, anthropic_body, candidates, upstream_header_val):
        success_stream = None
        json_resp = None
        dialled_id = None
        for m in candidates:
            raw_mid = m.get("id")
            clean_mid = raw_mid.removeprefix("openrouter/")
            dialled_id = f"openrouter/{clean_mid}"
            
            openai_body = anthropic_to_openai_body(anthropic_body, dialled_id)
            openai_body["stream"] = True
            
            try:
                req = urllib.request.Request(
                    gateway_url,
                    data=json.dumps(openai_body).encode("utf-8"),
                    headers=headers,
                    method="POST"
                )
                resp = urllib.request.urlopen(req, timeout=60)
                if resp.status == 200:
                    ct = resp.headers.get("Content-Type", "")
                    if "application/json" in ct and "text/event-stream" not in ct:
                        body_bytes = resp.read()
                        resp.close()
                        try:
                            json_resp = json.loads(body_bytes.decode("utf-8"))
                            json_resp["model"] = dialled_id
                            break
                        except Exception:
                            continue
                    else:
                        success_stream = resp
                        break
            except Exception:
                continue

        if not success_stream and not json_resp:
            self.send_error(502, "Upstream Anthropic stream failed")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        if upstream_header_val:
            self.send_header("x-makefun-upstream-failures", upstream_header_val)
        self.close_connection = True
        self.end_headers()

        if json_resp:
            choices = json_resp.get("choices", [])
            choice = choices[0] if choices else {}
            msg = choice.get("message", {})
            content_text = msg.get("content") or ""
            tool_calls = msg.get("tool_calls", [])
            finish_reason = choice.get("finish_reason")
            
            stop_reason = "end_turn"
            if finish_reason == "length":
                stop_reason = "max_tokens"
            elif finish_reason in ("tool_calls", "function_call") or tool_calls:
                stop_reason = "tool_use"

            usage = json_resp.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 10)
            output_tokens = usage.get("completion_tokens", 1)

            # 1. message_start
            msg_id = next_id("msg")
            start_event = {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": dialled_id,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 1}
                }
            }
            self.wfile.write(f"event: message_start\ndata: {json.dumps(start_event, separators=(',', ':'))}\n\n".encode("utf-8"))
            self.wfile.flush()

            # 2. content_block_start
            block_start = {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""}
            }
            self.wfile.write(f"event: content_block_start\ndata: {json.dumps(block_start, separators=(',', ':'))}\n\n".encode("utf-8"))
            self.wfile.flush()

            if content_text:
                delta_event = {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": content_text}
                }
                self.wfile.write(f"event: content_block_delta\ndata: {json.dumps(delta_event, separators=(',', ':'))}\n\n".encode("utf-8"))
                self.wfile.flush()

            for tc in tool_calls:
                func = tc.get("function", {})
                args_str = func.get("arguments", "{}")
                delta_event = {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": args_str}
                }
                self.wfile.write(f"event: content_block_delta\ndata: {json.dumps(delta_event, separators=(',', ':'))}\n\n".encode("utf-8"))
                self.wfile.flush()

            # 3. content_block_stop
            block_stop = {"type": "content_block_stop", "index": 0}
            self.wfile.write(f"event: content_block_stop\ndata: {json.dumps(block_stop, separators=(',', ':'))}\n\n".encode("utf-8"))
            self.wfile.flush()

            # 4. message_delta
            msg_delta = {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens}
            }
            self.wfile.write(f"event: message_delta\ndata: {json.dumps(msg_delta, separators=(',', ':'))}\n\n".encode("utf-8"))
            self.wfile.flush()

            # 5. message_stop
            msg_stop = {"type": "message_stop"}
            self.wfile.write(f"event: message_stop\ndata: {json.dumps(msg_stop, separators=(',', ':'))}\n\n".encode("utf-8"))
            self.wfile.flush()
            return

        msg_started = False
        block_started = False
        block_stopped = False
        delta_sent = False
        stop_sent = False
        output_tokens = 1
        stop_reason = "end_turn"

        try:
            # 1. message_start
            msg_id = next_id("msg")
            start_event = {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": dialled_id,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 1}
                }
            }
            self.wfile.write(f"event: message_start\ndata: {json.dumps(start_event, separators=(',', ':'))}\n\n".encode("utf-8"))
            self.wfile.flush()
            msg_started = True

            # 2. content_block_start
            block_start = {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""}
            }
            self.wfile.write(f"event: content_block_start\ndata: {json.dumps(block_start, separators=(',', ':'))}\n\n".encode("utf-8"))
            self.wfile.flush()
            block_started = True

            for line in success_stream:
                line_str = line.decode("utf-8", errors="ignore")
                if line_str.startswith("data: "):
                    data_part = line_str[6:].strip()
                    if data_part == "[DONE]":
                        break
                    try:
                        chunk_json = json.loads(data_part)
                        choices = chunk_json.get("choices", [])
                        if choices:
                            choice = choices[0]
                            delta = choice.get("delta", {})
                            content_piece = delta.get("content")
                            tool_calls = delta.get("tool_calls")
                            
                            if content_piece:
                                output_tokens += 1
                                delta_event = {
                                    "type": "content_block_delta",
                                    "index": 0,
                                    "delta": {"type": "text_delta", "text": content_piece}
                                }
                                self.wfile.write(f"event: content_block_delta\ndata: {json.dumps(delta_event, separators=(',', ':'))}\n\n".encode("utf-8"))
                                self.wfile.flush()
                            elif tool_calls:
                                for tc in tool_calls:
                                    func = tc.get("function", {})
                                    args_piece = func.get("arguments")
                                    if args_piece:
                                        output_tokens += 1
                                        delta_event = {
                                            "type": "content_block_delta",
                                            "index": 0,
                                            "delta": {"type": "input_json_delta", "partial_json": args_piece}
                                        }
                                        self.wfile.write(f"event: content_block_delta\ndata: {json.dumps(delta_event, separators=(',', ':'))}\n\n".encode("utf-8"))
                                        self.wfile.flush()

                            fr = choice.get("finish_reason")
                            if fr == "length":
                                stop_reason = "max_tokens"
                            elif fr in ("tool_calls", "function_call"):
                                stop_reason = "tool_use"
                    except Exception:
                        pass
        except Exception:
            pass
        finally:
            try:
                success_stream.close()
            except Exception:
                pass

            try:
                if block_started and not block_stopped:
                    block_stop = {"type": "content_block_stop", "index": 0}
                    self.wfile.write(f"event: content_block_stop\ndata: {json.dumps(block_stop, separators=(',', ':'))}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    block_stopped = True

                if msg_started and not delta_sent:
                    msg_delta = {
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                        "usage": {"output_tokens": output_tokens}
                    }
                    self.wfile.write(f"event: message_delta\ndata: {json.dumps(msg_delta, separators=(',', ':'))}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    delta_sent = True

                if msg_started and not stop_sent:
                    msg_stop = {"type": "message_stop"}
                    self.wfile.write(f"event: message_stop\ndata: {json.dumps(msg_stop, separators=(',', ':'))}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    stop_sent = True
            except Exception:
                pass

    def send_json_response(self, status_code, data, extra_headers=None):
        body = json.dumps(data, separators=(',', ':')).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if extra_headers:
            for k, v in extra_headers.items():
                if v:
                    self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)


# ==============================================================================
# SECTION 6: MAIN ENTRYPOINT & STARTUP
# ==============================================================================

def main():
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE")
    env_url = os.environ.get("MAKEFUN_ENV_URL")

    is_list_arg = "--list" in sys.argv
    
    # Rule 6: running with no MAKEFUN_PROMPT_FILE and no MAKEFUN_ENV_URL prints {"providers":[]} and exits 0.
    if is_list_arg or (not prompt_file and not env_url):
        models_list, _ = parse_brief()
        catalog = build_catalog(models_list)
        if not prompt_file or not models_list:
            catalog = {"providers": []}
        print(json.dumps(catalog, separators=(',', ':')))
        sys.exit(0)

    # Read brief and build catalog
    models_list, _ = parse_brief()
    catalog = build_catalog(models_list)

    # Pick free TCP port binding 0.0.0.0
    temp_sock = socket.socket()
    temp_sock.bind(("0.0.0.0", 0))
    port = temp_sock.getsockname()[1]
    temp_sock.close()

    # Start HTTP server on 0.0.0.0
    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), AggregatorHandler)

    # Register with env and start polling work
    register_and_poll(port, catalog)

    # Print catalog to stdout
    print(json.dumps(catalog, separators=(',', ':')))
    sys.stdout.flush()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
