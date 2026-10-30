// Drives the operator UI in a headless browser: draws an area, starts a mission,
// waits for survivors, then screenshots.
// Usage: node scripts/shot_running.cjs <url> <out.png> [mission_seconds]
const { chromium } = require("playwright");

(async () => {
  const url = process.argv[2] || "http://localhost:8000/";
  const out = process.argv[3] || "/tmp/shots/ui_running.png";
  const runFor = Number(process.argv[4] ?? 25) * 1000;
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  page.on("pageerror", (e) => console.log("PAGE ERROR:", e.message));
  // The MJPEG streams never go idle, so wait for the DOM and give the map time to load.
  await page.goto(url, { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(4000);

  const box = await page.locator(".map canvas").first().boundingBox();
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
  const r = Math.min(box.width, box.height) * 0.3;
  const click = async (x, y) => {
    await page.mouse.move(x, y);
    await page.waitForTimeout(90);
    await page.mouse.down();
    await page.waitForTimeout(90);
    await page.mouse.up();
    await page.waitForTimeout(200);
  };

  // Search area: four corners, then click the first vertex to close it.
  await page.click("button:has-text('Area')");
  await page.waitForTimeout(200);
  for (const [dx, dy] of [[-r, -r], [r, -r], [r, r], [-r, r]]) await click(cx + dx, cy + dy);
  await click(cx - r, cy - r);
  await page.waitForTimeout(400);

  // Last-seen point for the wilderness scenario.
  await page.click("button:has-text('LKP')");
  await page.waitForTimeout(200);
  await click(cx, cy);
  await page.waitForTimeout(300);

  // One operator priority region with the default weight.
  await page.click("button:has-text('Region')");
  await page.waitForTimeout(200);
  const rr = r * 0.45;
  for (const [dx, dy] of [[-rr, -rr], [rr, -rr], [rr, rr], [-rr, rr]]) await click(cx + dx, cy + dy);
  await click(cx - rr, cy - rr);
  await page.waitForTimeout(400);
  await page.click("button:has-text('Edit')");

  // Turn on the prior blend overlay if the server sent one.
  const priorToggle = page.locator(".prior-toggles label[title^=\"Overlay the\"] input").first();
  if (await priorToggle.count()) {
    await priorToggle.check({ force: true });
    await page.waitForTimeout(300);
  }

  await page.click("button:has-text('Start')");
  await page.waitForTimeout(runFor);

  await page.screenshot({ path: out });
  console.log("saved", out);
  await browser.close();
})().catch((e) => {
  console.error(e.message);
  process.exit(1);
});