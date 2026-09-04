#!/usr/bin/env node
/**
 * Phase 5 browser verification.
 *
 * Usage:
 *   node frontend/scripts/capture_phase5.mjs screenshot <image-path> <output-png>
 *
 * Requires Chrome to already be running with:
 *   /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
 *     --headless=new --remote-debugging-port=9222 --user-data-dir=/tmp/abyss-chrome
 */

import { writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const FRONTEND_URL = process.env.ABYSS_FRONTEND_URL ?? "http://127.0.0.1:5173";
const CDP_NEW_PAGE_URL = "http://127.0.0.1:9222/json/new";

class CdpClient {
  constructor(socket) {
    this.socket = socket;
    this.nextId = 1;
    this.pending = new Map();
    this.socket.onmessage = (event) => {
      const message = JSON.parse(event.data.toString());
      if (message.id && this.pending.has(message.id)) {
        const { resolve: ok, reject } = this.pending.get(message.id);
        this.pending.delete(message.id);
        if (message.error) reject(new Error(JSON.stringify(message.error)));
        else ok(message.result ?? {});
      }
    };
  }

  send(method, params = {}, timeoutMs = 30000) {
    const id = this.nextId++;
    const payload = JSON.stringify({ id, method, params });
    return new Promise((resolvePromise, rejectPromise) => {
      const timeout = setTimeout(() => {
        this.pending.delete(id);
        rejectPromise(new Error(`CDP command timed out: ${method}`));
      }, timeoutMs);
      this.pending.set(id, {
        resolve: (value) => {
          clearTimeout(timeout);
          resolvePromise(value);
        },
        reject: (error) => {
          clearTimeout(timeout);
          rejectPromise(error);
        },
      });
      this.socket.send(payload);
    });
  }
}

function sleep(ms) {
  return new Promise((resolvePromise) => setTimeout(resolvePromise, ms));
}

async function connect() {
  const target = await fetch(`${CDP_NEW_PAGE_URL}?${encodeURIComponent("about:blank")}`, {
    method: "PUT",
  }).then((response) => {
    if (!response.ok) throw new Error(`Chrome CDP is not reachable: ${response.status}`);
    return response.json();
  });
  const socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolvePromise, rejectPromise) => {
    socket.onopen = resolvePromise;
    socket.onerror = rejectPromise;
  });
  const client = new CdpClient(socket);
  await client.send("Page.enable");
  await client.send("DOM.enable");
  await client.send("Runtime.enable");
  return client;
}

async function waitForText(client, text, timeoutMs) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const result = await client.send("Runtime.evaluate", {
      expression: `document.body.innerText.includes(${JSON.stringify(text)})`,
      returnByValue: true,
    });
    if (result.result.value) return;
    await sleep(500);
  }
  throw new Error(`Timed out waiting for page text: ${text}`);
}

async function waitForExpression(client, expression, timeoutMs, label) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const result = await client.send("Runtime.evaluate", {
      expression,
      returnByValue: true,
    });
    if (result.result.value) return;
    await sleep(500);
  }
  throw new Error(`Timed out waiting for expression: ${label}`);
}

async function screenshot(imagePath, outputPng) {
  console.error("connecting_to_chrome");
  const client = await connect();
  console.error("connected_to_chrome");
  await client.send("Emulation.setDeviceMetricsOverride", {
    width: 1440,
    height: 1200,
    deviceScaleFactor: 1,
    mobile: false,
  });
  console.error("navigating_to_frontend");
  await client.send("Page.navigate", { url: `${FRONTEND_URL}?autodetect=1` });
  await waitForText(client, "ABYSS", 15000);
  await sleep(1000);
  console.error("setting_file_input");

  const documentResult = await client.send("DOM.getDocument", { depth: -1, pierce: true });
  const input = await client.send("DOM.querySelector", {
    nodeId: documentResult.root.nodeId,
    selector: "input[type=file]",
  });
  if (!input.nodeId) throw new Error("Could not find the file upload input");
  await client.send("DOM.setFileInputFiles", {
    nodeId: input.nodeId,
    files: [resolve(imagePath)],
  });

  console.error("waiting_for_real_api_output");
  await waitForExpression(
    client,
    `document.body.innerText.includes("simulated") || document.body.innerText.includes("Detections\\n0") || document.body.innerText.includes("Failed to fetch") || document.body.innerText.includes("Backend unavailable")`,
    180000,
    "priority/geolocation output, zero detections, or visible error state",
  );
  await sleep(2500);

  console.error("reading_visible_text");
  const visibleText = await client.send("Runtime.evaluate", {
    expression: "document.body.innerText",
    returnByValue: true,
  });
  console.error("capturing_screenshot");
  const capture = await client.send("Page.captureScreenshot", { format: "png", fromSurface: true });
  console.error("writing_screenshot");
  await writeFile(outputPng, Buffer.from(capture.data, "base64"));
  console.log(JSON.stringify({ imagePath: resolve(imagePath), outputPng: resolve(outputPng), visibleText: visibleText.result.value }, null, 2));
  process.exit(0);
}

const [command, imagePath, outputPng] = process.argv.slice(2);
if (command !== "screenshot" || !imagePath || !outputPng) {
  console.error("Usage: node frontend/scripts/capture_phase5.mjs screenshot <image-path> <output-png>");
  process.exit(2);
}

screenshot(imagePath, outputPng).catch((error) => {
  console.error(error);
  process.exit(1);
});
