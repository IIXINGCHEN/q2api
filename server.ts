import { Hono } from "hono";
import { cors } from "hono/cors";
import { serveStatic } from "hono/deno";
import {
  Account,
  AccountCreate,
  AccountUpdate,
  ClaudeRequest,
  ChatCompletionRequest
} from "./types.ts";
import * as db from "./db.ts";
import * as auth from "./auth.ts";
import { convertClaudeToAmazonQRequest, convertOpenAIRequestToAmazonQ } from "./converter.ts";
import { sendChatRequest } from "./amazon_q.ts";
import { ClaudeStreamHandler } from "./stream_handler.ts";
// Deno KV是内置模块，不需要导入

const app = new Hono();

app.use("*", cors());

// --- Configuration ---
const ALLOWED_API_KEYS = (Deno.env.get("OPENAI_KEYS") || "")
  .split(",")
  .map(k => k.trim())
  .filter(k => k);

const WEB_PASSWORD = Deno.env.get("WEB_PASSWORD");
const MAX_ERROR_COUNT = parseInt(Deno.env.get("MAX_ERROR_COUNT") || "100");
const CONSOLE_ENABLED = (Deno.env.get("ENABLE_CONSOLE") || "true").toLowerCase() !== "false";

// 使用Deno KV进行会话存储
let kv: Deno.Kv;

// 初始化KV数据库
async function initKV() {
  if (!kv) {
    kv = await Deno.openKv();
  }
  return kv;
}

// --- Web Auth Helpers ---
function generateSessionToken(): string {
  return crypto.randomUUID();
}

async function createSession(): Promise<{ token: string; expires: number }> {
  const token = generateSessionToken();
  const expires = Date.now() + (24 * 60 * 60 * 1000); // 24 hours

  // 确保KV已初始化
  if (!kv) {
    kv = await Deno.openKv();
  }

  // 使用Deno KV存储会话
  await kv.set(["sessions", token], { expires }, { ttl: 24 * 60 * 60 * 1000 });

  return { token, expires };
}

async function validateSession(token: string): Promise<boolean> {
  // 确保KV已初始化
  if (!kv) {
    kv = await Deno.openKv();
  }

  const result = await kv.get(["sessions", token]);
  if (!result.value) return false;

  const session = result.value as { expires: number };
  if (Date.now() > session.expires) {
    await kv.delete(["sessions", token]);
    return false;
  }

  return true;
}

async function cleanupExpiredSessions() {
  // 使用Deno KV的TTL机制，过期的会话会自动清理
  // 不需要手动清理，但这个函数可以用于其他清理逻辑
  console.log("会话清理任务执行中（Deno KV自动处理过期会话）");
}

// 清理过期会话的定时任务（每小时执行一次）
Deno.cron("Cleanup Sessions", "0 * * * *", cleanupExpiredSessions);

// --- Web Auth Middleware ---
async function requireWebAuth(c: any, next: any) {
  // If no web password is set, allow access
  if (!WEB_PASSWORD) {
    return next();
  }

  const sessionToken = c.req.header("Cookie")?.match(/session=([^;]+)/)?.[1];

  if (!sessionToken || !(await validateSession(sessionToken))) {
    // Redirect to login page or return unauthorized for API calls
    if (c.req.path.startsWith("/api/")) {
      return c.json({ error: "Unauthorized" }, 401);
    } else {
      return c.redirect("/login");
    }
  }

  return next();
}

// --- Helpers ---

/**
 * 对敏感信息进行脱敏处理
 * @param account 账户信息
 * @returns 脱敏后的账户信息
 */
function sanitizeAccount(account: Account): Omit<Account, 'clientSecret' | 'refreshToken' | 'accessToken'> & {
  clientSecret: string;
  refreshToken?: string;
  accessToken?: string;
} {
  const sanitized = { ...account };
  
  // 对 clientSecret 进行脱敏，只显示前8位和后4位
  if (sanitized.clientSecret) {
    if (sanitized.clientSecret.length > 12) {
      sanitized.clientSecret = sanitized.clientSecret.substring(0, 8) + '***' + sanitized.clientSecret.substring(sanitized.clientSecret.length - 4);
    } else {
      sanitized.clientSecret = '***';
    }
  }
  
  // 对 refreshToken 进行脱敏，只显示前8位和后4位
  if (sanitized.refreshToken) {
    if (sanitized.refreshToken.length > 12) {
      sanitized.refreshToken = sanitized.refreshToken.substring(0, 8) + '***' + sanitized.refreshToken.substring(sanitized.refreshToken.length - 4);
    } else {
      sanitized.refreshToken = '***';
    }
  }
  
  // 对 accessToken 进行脱敏，只显示前8位和后4位
  if (sanitized.accessToken) {
    if (sanitized.accessToken.length > 12) {
      sanitized.accessToken = sanitized.accessToken.substring(0, 8) + '***' + sanitized.accessToken.substring(sanitized.accessToken.length - 4);
    } else {
      sanitized.accessToken = '***';
    }
  }
  
  return sanitized;
}

function extractTextFromEvent(payload: any): string {
  if (!payload || typeof payload !== 'object') return "";
  
  // 1. Check nested content in specific keys
  const keysToCheck = ["assistantResponseEvent", "assistantMessage", "message", "delta", "data"];
  for (const key of keysToCheck) {
    if (payload[key] && typeof payload[key] === 'object') {
      const inner = payload[key];
      if (inner.content && typeof inner.content === 'string') {
        return inner.content;
      }
    }
  }

  // 2. Check top-level content (string)
  if (payload.content && typeof payload.content === 'string') {
    return payload.content;
  }

  // 3. Check lists (chunks or content)
  const listKeys = ["chunks", "content"];
  for (const listKey of listKeys) {
    if (Array.isArray(payload[listKey])) {
      const parts = payload[listKey].map((item: any) => {
        if (typeof item === 'string') return item;
        if (typeof item === 'object') {
          if (item.content && typeof item.content === 'string') return item.content;
          if (item.text && typeof item.text === 'string') return item.text;
        }
        return "";
      });
      const joined = parts.join("");
      if (joined) return joined;
    }
  }
  
  // 4. Fallback: check text/delta/payload keys if they are strings
  const fallbackKeys = ["text", "delta", "payload"];
  for (const k of fallbackKeys) {
    if (payload[k] && typeof payload[k] === 'string') {
      return payload[k];
    }
  }
  
  return "";
}

async function refreshAccessTokenInDb(accountId: string): Promise<Account> {
  const acc = await db.getAccount(accountId);
  if (!acc) throw new Error("Account not found");

  if (!acc.clientId || !acc.clientSecret || !acc.refreshToken) {
    throw new Error("Account missing credentials for refresh");
  }

  try {
    const data = await auth.refreshToken(acc.clientId, acc.clientSecret, acc.refreshToken);
    const newAccess = data.accessToken;
    const newRefresh = data.refreshToken || acc.refreshToken;
    
    await db.updateAccountTokens(accountId, newAccess, newRefresh, "success");
    
    const updated = await db.getAccount(accountId);
    return updated!;
  } catch (e: any) {
    await db.updateAccountRefreshStatus(accountId, "failed");
    throw e;
  }
}

async function resolveAccountForKey(bearerKey?: string): Promise<Account> {
  if (ALLOWED_API_KEYS.length > 0) {
    if (!bearerKey || !ALLOWED_API_KEYS.includes(bearerKey)) {
      throw new Error("Invalid or missing API key"); // 401
    }
  }

  const candidates = await db.listAccounts(true);
  if (candidates.length === 0) {
    throw new Error("No enabled account available"); // 401
  }

  // 智能账户选择：优先选择错误率低且最近成功的账户
  const scoredAccounts = candidates.map(account => {
    let score = 0;

    // 基础分数：成功请求越多分数越高
    score += account.success_count * 10;

    // 错误惩罚：每个错误减5分
    score -= account.error_count * 5;

    // 最近更新时间奖励：最近更新的账户分数更高
    const lastUpdate = new Date(account.updated_at).getTime();
    const now = Date.now();
    const hoursSinceUpdate = (now - lastUpdate) / (1000 * 60 * 60);
    score += Math.max(0, 24 - hoursSinceUpdate); // 24小时内更新的有奖励

    return { account, score };
  });

  // 按分数排序，选择分数最高的账户
  scoredAccounts.sort((a, b) => b.score - a.score);

  // 从高分到低分尝试找到可用账户
  for (const { account } of scoredAccounts) {
    if (isAccountAvailable(account)) {
      return account;
    }
  }

  // 如果所有账户都不可用，返回分数最高的账户（让调用方处理错误）
  return scoredAccounts[0].account;
}

// 检查账户是否可用（基于错误率和限流状态）
function isAccountAvailable(account: Account): boolean {
  // 错误率检查：如果错误次数过多，暂时禁用
  if (account.error_count >= 10) {
    return false;
  }

  // 总成功率检查：成功率太低的账户优先级降低
  const totalRequests = account.success_count + account.error_count;
  if (totalRequests >= 20 && account.success_count / totalRequests < 0.3) {
    return false;
  }

  return true;
}

// 检测是否为限流错误
function isRateLimitError(error: any): boolean {
  if (typeof error === 'string') {
    return error.includes('429') ||
           error.includes('ThrottlingException') ||
           error.includes('Too many requests');
  }
  return false;
}

// 智能重试的请求发送函数
async function sendRequestWithRetry(
  req: ClaudeRequest | ChatCompletionRequest,
  aqRequest: Record<string, any>,
  bearerKey?: string,
  maxRetries: number = 3
): Promise<{
  eventStream: AsyncGenerator<[string, any], void, unknown>
  account: Account
}> {
  const attemptedAccounts = new Set<string>();

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    let account: Account;

    try {
      // 选择账户，排除已经尝试过的账户
      account = await resolveAccountForRetry(bearerKey, attemptedAccounts);
      attemptedAccounts.add(account.id);

      console.log(`尝试使用账户 ${account.label || account.id} (第${attempt + 1}次尝试)`);

      let access = account.accessToken;
      if (!access) {
        account = await refreshAccessTokenInDb(account.id);
        access = account.accessToken;
      }
      if (!access) throw new Error("Access token missing");

      // 发送请求
      const result = await sendChatRequest(access, aqRequest);

      // 成功后更新账户统计并返回
      // 统计已由 sendRequestWithRetry 处理
      return { eventStream: result.eventStream, account };

    } catch (e: any) {
      console.error(`账户 ${attemptedAccounts[attemptedAccounts.size - 1]} 请求失败:`, e.message);

      // 检查是否是限流错误
      if (isRateLimitError(e.message)) {
        console.log('检测到限流错误，尝试切换账户...');

        // 如果没有更多账户可尝试，抛出限流错误
        const availableAccounts = (await db.listAccounts(true)).filter(acc => !attemptedAccounts.has(acc.id));
        if (availableAccounts.length === 0) {
          throw new Error(`所有账户都遇到了限流错误: ${e.message}`);
        }

        continue; // 尝试下一个账户
      }

      // 非限流错误，直接抛出
      throw e;
    }
  }

  throw new Error(`在 ${maxRetries} 次尝试后仍然无法处理请求`);
}

// 用于重试的账户选择函数
async function resolveAccountForRetry(bearerKey?: string, attemptedAccounts: Set<string> = new Set()): Promise<Account> {
  if (ALLOWED_API_KEYS.length > 0) {
    if (!bearerKey || !ALLOWED_API_KEYS.includes(bearerKey)) {
      throw new Error("Invalid or missing API key");
    }
  }

  const candidates = await db.listAccounts(true);
  const availableCandidates = candidates.filter(account => !attemptedAccounts.has(account.id));

  if (availableCandidates.length === 0) {
    throw new Error("没有更多可用的账户进行重试");
  }

  // 使用相同的智能选择算法
  const scoredAccounts = availableCandidates.map(account => {
    let score = 0;
    score += account.success_count * 10;
    score -= account.error_count * 5;

    const lastUpdate = new Date(account.updated_at).getTime();
    const now = Date.now();
    const hoursSinceUpdate = (now - lastUpdate) / (1000 * 60 * 60);
    score += Math.max(0, 24 - hoursSinceUpdate);

    return { account, score };
  });

  scoredAccounts.sort((a, b) => b.score - a.score);

  // 返回分数最高且可用的账户
  for (const { account } of scoredAccounts) {
    if (isAccountAvailable(account)) {
      return account;
    }
  }

  return scoredAccounts[0].account;
}

function extractBearer(authHeader?: string): string | undefined {
  if (!authHeader) return undefined;
  if (authHeader.startsWith("Bearer ")) return authHeader.substring(7).trim();
  return authHeader.trim();
}

// --- Background Tasks ---

async function refreshStaleTokens() {
  try {
    const accounts = await db.listAccounts(true);
    const now = Date.now() / 1000;
    
    for (const acc of accounts) {
      let shouldRefresh = false;
      if (!acc.last_refresh_time || acc.last_refresh_status === "never") {
        shouldRefresh = true;
      } else {
        try {
          const lastTime = new Date(acc.last_refresh_time).getTime() / 1000;
          if (now - lastTime > 1500) { // 25 mins
            shouldRefresh = true;
          }
        } catch {
          shouldRefresh = true;
        }
      }

      if (shouldRefresh) {
        try {
          await refreshAccessTokenInDb(acc.id);
          console.log(`Refreshed token for ${acc.id}`);
        } catch (e) {
          console.error(`Failed to refresh ${acc.id}:`, e);
        }
      }
    }
  } catch (e) {
    console.error("Error in refreshStaleTokens:", e);
  }
}

// Deno Cron (works in Deploy)
Deno.cron("Refresh Tokens", "*/5 * * * *", refreshStaleTokens);

// --- Routes ---

app.get("/healthz", (c) => c.json({ status: "ok" }));

// Login page
app.get("/login", serveStatic({ path: "./frontend/login.html" }));

// Login API
app.post("/api/login", async (c) => {
  if (!WEB_PASSWORD) {
    return c.json({ error: "Web password not configured" }, 500);
  }

  const body = await c.req.json();
  const { password } = body;

  if (password === WEB_PASSWORD) {
    const { token } = await createSession();
    return c.json({ success: true, token });
  } else {
    return c.json({ error: "Invalid password" }, 401);
  }
});

// Logout API
app.post("/api/logout", async (c) => {
  const sessionToken = c.req.header("Cookie")?.match(/session=([^;]+)/)?.[1];
  if (sessionToken) {
    // 确保KV已初始化
    if (!kv) {
      kv = await Deno.openKv();
    }
    await kv.delete(["sessions", sessionToken]);
  }
  return c.json({ success: true });
});

// Frontend with auth protection
app.get("/", async (c) => {
  return await requireWebAuth(c, async () => {
    return serveStatic({ path: "./frontend/index.html" })(c);
  });
});

// Protect all frontend routes
app.get("/frontend/*", async (c) => {
  return await requireWebAuth(c, async () => {
    return serveStatic({ path: "./frontend" })(c);
  });
});

// Account Management
if (CONSOLE_ENABLED) {
  app.get("/v2/accounts", async (c) => {
    const accounts = await db.listAccounts();
    const sanitizedAccounts = accounts.map(sanitizeAccount);
    return c.json(sanitizedAccounts);
  });

  app.post("/v2/accounts", async (c) => {
    const body = await c.req.json<AccountCreate>();
    const acc = await db.createAccount(body);
    return c.json(sanitizeAccount(acc));
  });

  app.get("/v2/accounts/:id", async (c) => {
    const id = c.req.param("id");
    const acc = await db.getAccount(id);
    if (!acc) return c.json({ error: "Not found" }, 404);
    return c.json(sanitizeAccount(acc));
  });

  app.delete("/v2/accounts/:id", async (c) => {
    const id = c.req.param("id");
    const deleted = await db.deleteAccount(id);
    if (!deleted) return c.json({ error: "Not found" }, 404);
    return c.json({ deleted: id });
  });

  app.patch("/v2/accounts/:id", async (c) => {
    const id = c.req.param("id");
    const body = await c.req.json<AccountUpdate>();
    const updated = await db.updateAccount(id, body);
    if (!updated) return c.json({ error: "Not found" }, 404);
    return c.json(updated);
  });

  app.post("/v2/accounts/:id/refresh", async (c) => {
    const id = c.req.param("id");
    try {
      const acc = await refreshAccessTokenInDb(id);
      return c.json(sanitizeAccount(acc));
    } catch (e: any) {
      return c.json({ error: e.message }, 502);
    }
  });

  // 获取账户状态统计
  app.get("/v2/accounts/stats", async (c) => {
    const accounts = await db.listAccounts();

    const stats = {
      total: accounts.length,
      enabled: accounts.filter(acc => acc.enabled).length,
      disabled: accounts.filter(acc => !acc.enabled).length,
      accounts: accounts.map(account => {
        const totalRequests = account.success_count + account.error_count;
        const successRate = totalRequests > 0 ? (account.success_count / totalRequests * 100).toFixed(2) : "0.00";

        return {
          id: account.id,
          label: account.label,
          enabled: account.enabled,
          success_count: account.success_count,
          error_count: account.error_count,
          success_rate: `${successRate}%`,
          last_refresh_status: account.last_refresh_status,
          last_refresh_time: account.last_refresh_time,
          created_at: account.created_at,
          updated_at: account.updated_at,
          // 脱敏显示clientId的前8位
          client_id_prefix: account.clientId ? account.clientId.substring(0, 8) + "***" : null
        };
      })
    };

    return c.json(stats);
  });

  // 重置账户错误计数
  app.post("/v2/accounts/:id/reset-errors", async (c) => {
    const id = c.req.param("id");
    const account = await db.getAccount(id);

    if (!account) {
      return c.json({ error: "Account not found" }, 404);
    }

    // 重置错误计数并重新启用账户
    await db.updateAccountStats(id, true, 1000); // 使用高maxErrorCount确保不会禁用
    const updated = await db.updateAccount(id, { enabled: true });

    return c.json({ message: "Error count reset successfully", account: sanitizeAccount(updated!) });
  });

  // Device Auth Flow
  const AUTH_SESSIONS = new Map<string, any>();
  app.post("/v2/auth/start", async (c) => {
    const body = await c.req.json<{label?: string, enabled?: boolean}>();
    try {
      const [cid, csec] = await auth.registerClientMin();
      const dev = await auth.deviceAuthorize(cid, csec);

      const authId = crypto.randomUUID();
      const sess = {
        clientId: cid,
        clientSecret: csec,
        deviceCode: dev.deviceCode,
        interval: dev.interval || 1,
        expiresIn: dev.expiresIn || 600,
        verificationUriComplete: dev.verificationUriComplete,
        userCode: dev.userCode,
        startTime: Math.floor(Date.now() / 1000),
        label: body.label,
        enabled: body.enabled !== false,
        status: "pending",
        error: null,
        accountId: null
      };
      AUTH_SESSIONS.set(authId, sess);

      return c.json({
        authId,
        verificationUriComplete: sess.verificationUriComplete,
        userCode: sess.userCode,
        expiresIn: sess.expiresIn,
        interval: sess.interval
      });
    } catch (e: any) {
      return c.json({ error: e.message }, 502);
    }
  });

  app.get("/v2/auth/status/:authId", (c) => {
    const authId = c.req.param("authId");
    const sess = AUTH_SESSIONS.get(authId);
    if (!sess) return c.json({ error: "Not found" }, 404);

    const now = Math.floor(Date.now() / 1000);
    const deadline = sess.startTime + Math.min(sess.expiresIn, 300); // 5 min cap
    const remaining = Math.max(0, deadline - now);

    return c.json({
      status: sess.status,
      remaining,
      error: sess.error,
      accountId: sess.accountId
    });
  });

  app.post("/v2/auth/claim/:authId", async (c) => {
    const authId = c.req.param("authId");
    const sess = AUTH_SESSIONS.get(authId);
    if (!sess) return c.json({ error: "Not found" }, 404);

    if (["completed", "timeout", "error"].includes(sess.status)) {
      return c.json({
        status: sess.status,
        accountId: sess.accountId,
        error: sess.error
      });
    }

    try {
      const toks = await auth.pollTokenDeviceCode(
        sess.clientId,
        sess.clientSecret,
        sess.deviceCode,
        sess.interval,
        sess.expiresIn,
        300
      );

      const acc = await db.createAccount({
        clientId: sess.clientId,
        clientSecret: sess.clientSecret,
        accessToken: toks.accessToken,
        refreshToken: toks.refreshToken,
        label: sess.label,
        enabled: sess.enabled
      });

      sess.status = "completed";
      sess.accountId = acc.id;

      return c.json({
        status: "completed",
        account: sanitizeAccount(acc)
      });
    } catch (e: any) {
      if (e.message.includes("timeout")) {
        sess.status = "timeout";
        return c.json({ error: "Timeout" }, 408);
      } else {
        sess.status = "error";
        sess.error = e.message;
        return c.json({ error: e.message }, 502);
      }
    }
  });
}

// Chat API

app.post("/v1/messages", async (c) => {
    const req = await c.req.json<ClaudeRequest>();
    const authHeader = c.req.header("Authorization");
    const bearer = extractBearer(authHeader);

    try {
        // Convert request
        const aqRequest = convertClaudeToAmazonQRequest(req);

        // 使用智能重试机制发送请求
        const { eventStream, account } = await sendRequestWithRetry(req, aqRequest, bearer);

        console.log(`成功使用账户: ${account.label || account.id} 处理请求`);
        
        if (req.stream) {
            const stream = new ReadableStream({
                async start(controller) {
                    const handler = new ClaudeStreamHandler(req.model, 0);
                    const encoder = new TextEncoder();
                    
                    try {
                        for await (const [eventType, payload] of eventStream) {
                            for await (const sse of handler.handleEvent(eventType, payload)) {
                                controller.enqueue(encoder.encode(sse));
                            }
                        }
                        for await (const sse of handler.finish()) {
                            controller.enqueue(encoder.encode(sse));
                        }
                        // 统计已由 sendRequestWithRetry 处理
                    } catch (e) {
                        // 统计已由 sendRequestWithRetry 处理
                        console.error("Stream error:", e);
                        controller.error(e);
                    } finally {
                        controller.close();
                    }
                }
            });

            return new Response(stream, {
                headers: {
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            });
        } else {
            // Non-streaming: accumulate response
            const handler = new ClaudeStreamHandler(req.model, 0);
            const contentBlocks: any[] = [];
            let usage = { input_tokens: 0, output_tokens: 0 };
            let stopReason = null;
            
            try {
                for await (const [eventType, payload] of eventStream) {
                    for await (const sse of handler.handleEvent(eventType, payload)) {
                        if (sse.startsWith("event: ")) {
                            const lines = sse.split("\n");
                            const dataLine = lines.find(l => l.startsWith("data: "));
                            if (dataLine) {
                                const data = JSON.parse(dataLine.substring(6));
                                const dtype = data.type;
                                if (dtype === "content_block_start") {
                                    const idx = data.index;
                                    while (contentBlocks.length <= idx) contentBlocks.push(null);
                                    contentBlocks[idx] = data.content_block;
                                } else if (dtype === "content_block_delta") {
                                    const idx = data.index;
                                    const delta = data.delta;
                                    if (contentBlocks[idx]) {
                                        if (delta.type === "text_delta") {
                                            contentBlocks[idx].text = (contentBlocks[idx].text || "") + delta.text;
                                        } else if (delta.type === "input_json_delta") {
                                            contentBlocks[idx].partial_json = (contentBlocks[idx].partial_json || "") + delta.partial_json;
                                        }
                                    }
                                } else if (dtype === "content_block_stop") {
                                    const idx = data.index;
                                    if (contentBlocks[idx]?.type === "tool_use" && contentBlocks[idx].partial_json) {
                                        try {
                                            contentBlocks[idx].input = JSON.parse(contentBlocks[idx].partial_json);
                                            delete contentBlocks[idx].partial_json;
                                        } catch {}
                                    }
                                } else if (dtype === "message_delta") {
                                    usage = data.usage || usage;
                                    stopReason = data.delta?.stop_reason;
                                }
                            }
                        }
                    }
                }
                for await (const sse of handler.finish()) {
                    if (sse.startsWith("event: message_delta")) {
                        const lines = sse.split("\n");
                        const dataLine = lines.find(l => l.startsWith("data: "));
                        if (dataLine) {
                            const data = JSON.parse(dataLine.substring(6));
                            usage = data.usage || usage;
                            stopReason = data.delta?.stop_reason;
                        }
                    }
                }
                // 统计已由 sendRequestWithRetry 处理
            } catch (e) {
                // 统计已由 sendRequestWithRetry 处理
                throw e;
            }
            
            return c.json({
                id: `msg_${crypto.randomUUID()}`,
                type: "message",
                role: "assistant",
                model: req.model,
                content: contentBlocks.filter(b => b !== null),
                stop_reason: stopReason,
                stop_sequence: null,
                usage: usage
            });
        }

    } catch (e: any) {
        // 统计已由 sendRequestWithRetry 处理
        return c.json({ error: e.message }, 502);
    }
});

app.post("/v1/chat/completions", async (c) => {
    const req = await c.req.json<ChatCompletionRequest>();
    const authHeader = c.req.header("Authorization");
    const bearer = extractBearer(authHeader);

    const aqRequest = convertOpenAIRequestToAmazonQ(req);
    const doStream = req.stream === true;
    const model = req.model || "unknown";
    const created = Math.floor(Date.now() / 1000);
    const id = `chatcmpl-${crypto.randomUUID()}`;

    try {
        // 使用智能重试机制发送请求
        const { eventStream, account } = await sendRequestWithRetry(req, aqRequest, bearer);

        console.log(`成功使用账户: ${account.label || account.id} 处理请求`);

        if (doStream) {
            const stream = new ReadableStream({
                async start(controller) {
                    const encoder = new TextEncoder();
                    const sendSSE = (data: any) => {
                        controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
                    };

                    try {
                        // Initial chunk
                        sendSSE({
                            id, object: "chat.completion.chunk", created, model,
                            choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }]
                        });

                        for await (const [eventType, payload] of eventStream) {
                            const text = extractTextFromEvent(payload);
                            if (text) {
                                sendSSE({
                                    id, object: "chat.completion.chunk", created, model,
                                    choices: [{ index: 0, delta: { content: text }, finish_reason: null }]
                                });
                            }
                        }
                        controller.enqueue(encoder.encode("data: [DONE]\n\n"));

                        // 统计已由 sendRequestWithRetry 处理
                    } catch (e) {
                        // 统计已由 sendRequestWithRetry 处理
                        console.error("Stream error:", e);
                        // Try to send error in stream if possible, or just close
                        const errObj = { error: { message: String(e), type: "server_error" } };
                         controller.enqueue(encoder.encode(`data: ${JSON.stringify(errObj)}\n\n`));
                    } finally {
                        controller.close();
                    }
                }
            });

            return new Response(stream, {
                headers: {
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            });
        } else {
            // Non-streaming: accumulate
            const chunks: string[] = [];
            for await (const [_, payload] of result.eventStream) {
                const text = extractTextFromEvent(payload);
                if (text) chunks.push(text);
            }
            
            if (chunks.length === 0) {
                 console.warn("No content chunks received from upstream.");
                 // If we have payload but no text, it might be an error message or empty response
            }

            const fullText = chunks.join("");
            // 统计已由 sendRequestWithRetry 处理

            return c.json({
                id,
                object: "chat.completion",
                created,
                model,
                choices: [{
                    index: 0,
                    message: { role: "assistant", content: fullText },
                    finish_reason: "stop"
                }],
                usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
            });
        }

    } catch (e: any) {
        // 统计已由 sendRequestWithRetry 处理
        return c.json({ error: e.message }, 502);
    }
});

Deno.serve(app.fetch);
