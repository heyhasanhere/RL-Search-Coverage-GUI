// Selects a drone in the list and screenshots the camera panel switching to it.
// Usage: node scripts/shot_select.cjs <url> <out.png>
const { chromium } = require("playwright");

(async () => {
  const url = process.argv[2] || "http://localhost:8000/";
  const out = process.argv[3] || "/tmp/shots/ui_select.png";
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  page.on("pageerror", (e) => console.log("PAGE ERROR:", e.message));
  await page.goto(url, { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(4000);

  // Click the third drone card (R1) and confirm the camera title follows it.
  const cards = page.locator(".card.drone");
  const n = await cards.count();
  console.log("drone cards:", n);
  await cards.nth(n - 1).click();
  await page.waitForTimeout(2500);

  const title = await page.locator(".cell-cam .video-title").innerText();
  const follow = await page.locator(".cell-3d .video-title").innerText();
  console.log("camera panel:", title);
  console.log("3D panel:", follow);
  console.log("selected cards:", await page.locator(".card.drone.sel").count());

  await page.screenshot({ path: out });
  console.log("saved", out);
  await browser.close();
})().catch((e) => {
  console.error(e.message);
  process.exit(1);
});