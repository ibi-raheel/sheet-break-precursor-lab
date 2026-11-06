import { chromium } from 'playwright'
const out = process.argv[2]
const b = await chromium.launch({ executablePath: process.env.PW_EXE }); const errs = []
const p = await b.newPage({ viewport: { width: 1440, height: 900 } })
p.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()) })
p.on('pageerror', (e) => errs.push(String(e)))
await p.goto('http://localhost:4791/'); await p.waitForTimeout(500)
await p.screenshot({ path: `${out}/01-horizons.png` })
await p.locator('h2').nth(1).scrollIntoViewIfNeeded(); await p.evaluate(() => document.querySelectorAll('h2')[1].scrollIntoView())
await p.screenshot({ path: `${out}/02-detectors.png` })
await p.evaluate(() => document.querySelectorAll('h2')[2].scrollIntoView())
await p.screenshot({ path: `${out}/03-shap-vs-lift.png` })
const m = await b.newPage({ viewport: { width: 375, height: 812 } })
await m.goto('http://localhost:4791/'); const sw = await m.evaluate(() => document.documentElement.scrollWidth)
console.log('mobile scrollWidth', sw, 'errors', errs)
await b.close()
