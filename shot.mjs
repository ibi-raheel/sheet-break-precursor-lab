// Usage: node shot.mjs <url> <outdir> [full]
import { chromium } from 'playwright'
const url = process.argv[2], out = process.argv[3], full = process.argv[4] === 'full'
const b = await chromium.launch(); const errs = []
for (const [w, h, tag] of [[1440, 900, 'd'], [375, 812, 'm']]) {
  const p = await b.newPage({ viewport: { width: w, height: h } })
  p.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') errs.push(`${tag}: ${m.text()}`) })
  p.on('pageerror', (e) => errs.push(`${tag}: ${e}`))
  await p.goto(url); await p.waitForTimeout(800)
  const sw = await p.evaluate(() => document.documentElement.scrollWidth)
  console.log(tag, 'scrollWidth', sw, 'viewport', w)
  if (full) await p.screenshot({ path: `${out}/${tag}-full.png`, fullPage: true })
  await p.close()
}
console.log('errors', errs)
await b.close()
