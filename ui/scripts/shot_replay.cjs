// Opens the replay mode with a packed (video) replay, plays it and checks that
// the recorded 3D view follows the slider, then screenshots.
// Usage: node scripts/shot_replay.cjs <url> <out.png> <replay id>
// Needs a browser with H.264 (the Playwright Chromium build has none):
// BROWSER=<path to a Chrome-family binary>, default Brave.
const { chromium } = require("playwright");

(async () => {
  const url = process.argv[2] || "http://localhost:8000/";
  const out = process.argv[3] || "/tmp/shots/ui_replay.png";
  const id = process.argv[4];
  const browser = await chromium.launch({
    executablePath: process.env.BROWSER || "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
  });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  page.on("pageerror", (e) => console.log("PAGE ERROR:", e.message));
  await page.goto(url, { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(2000);
  await page.getByRole("button", { name: "Replay", exact: true }).click();
  await page.waitForTimeout(1000);
  await page.locator(".replay-controls-panel select").selectOption(id);
  await page.waitForSelector(".cell-3d video");
  await page.waitForFunction(() => document.querySelector(".cell-3d video").readyState >= 1);

  const probe = () => page.evaluate(() => {
    const v = document.querySelector(".cell-3d video");
    const s = document.querySelector(".replay-slider");
    return { slider: Number(s.value), video: v.currentTime, rate: v.playbackRate, paused: v.paused, err: v.error && v.error.code };
  });
  // Scrub: the paused video must jump to the matching frame.
  await page.locator(".replay-slider").fill("640");
  await page.waitForTimeout(800);
  console.log("scrub", JSON.stringify(await probe()));
  await page.locator(".replay-controls-panel").getByRole("button", { name: "16x" }).click();
  await page.locator(".replay-controls-panel button.ok").click();
  await page.waitForTimeout(4000);
  console.log("play ", JSON.stringify(await probe()));
  // The overview and the selected drone's camera are recorded views too.
  await page.locator(".view-toggle").getByRole("button", { name: "Overview" }).click();
  await page.waitForTimeout(2500);
  console.log("views", JSON.stringify(await page.evaluate(() => [...document.querySelectorAll("video")].map((v) => ({
    src: v.getAttribute("src"), t: v.currentTime, ready: v.readyState, err: v.error && v.error.code })))));
  await page.screenshot({ path: out });
  await browser.close();
})();
