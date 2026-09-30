// Browser QA for the generated, self-contained historical placement explorer.
// Run with Playwright available through NODE_PATH or a local npm installation.
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

(async () => {
  const root = path.resolve(__dirname, '..');
  const qa = path.join(root, 'writeup/interactive/qa');
  fs.mkdirSync(qa, {recursive: true});
  const browser = await chromium.launch({headless: true, channel: process.env.PLAYWRIGHT_CHANNEL || undefined});
  const checks = [];
  try {
    for (const width of [736, 360]) {
      for (const colorScheme of ['light', 'dark']) {
        const page = await browser.newPage({viewport:{width,height:1000},colorScheme});
        const errors = [];
        page.on('pageerror', error => errors.push(String(error)));
        await page.goto('file://' + path.join(root,'writeup/interactive/placement-comparison.html'));
        const frame = page.frameLocator('iframe');
        await frame.locator('rect.tp-macro').first().waitFor();
        if (await frame.locator('[data-benchmark] option').count() !== 17) throw Error('Missing designs');
        for (const name of ['ibm01', 'ibm17', 'ibm18']) {
          await frame.locator('[data-benchmark]').selectOption(name);
          await frame.locator('[data-macro]').selectOption('0');
          await frame.locator('#tp-divider').focus();
          await frame.locator('#tp-divider').press('End');
          const state = await frame.locator('#thinkorplace-placement-comparison').evaluate(element => ({
            fits: element.dataset.labelsFit,
            width: element.getBoundingClientRect().width,
            scrollWidth: document.documentElement.scrollWidth,
            viewportWidth: window.innerWidth,
            count: element.querySelectorAll('rect.tp-macro').length,
            detail: element.querySelector('[data-detail]').textContent,
            divider: element.querySelector('#tp-divider').value,
            selected: element.querySelectorAll('[data-movement]').length,
          }));
          if (state.scrollWidth > state.viewportWidth || state.divider !== '100' || state.selected !== 1)
            throw Error(JSON.stringify(state));
          checks.push({outerViewportWidth:width,colorScheme,name,...state});
        }
        await frame.locator('#tp-divider').press('Home');
        await frame.locator('#tp-divider').press('ArrowRight');
        await frame.locator('#thinkorplace-placement-comparison').screenshot({path:path.join(qa,`${width}-${colorScheme}.png`)});
        if (errors.length) throw Error(errors.join('\n'));
        await page.close();
      }
    }
    fs.writeFileSync(path.join(qa,'checks.json'),JSON.stringify(checks,null,2)+'\n');
    console.log(`Passed ${checks.length} design/viewport/theme checks with keyboard interaction.`);
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error);process.exitCode=1;});
