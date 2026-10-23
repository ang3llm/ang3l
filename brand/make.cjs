// Screenshots the canvases in make.html into PNGs. Needs playwright-core and a Chromium (EXE env var).
const { chromium } = require(process.env.PW || 'playwright-core');
(async () => {
  const b = await chromium.launch({ executablePath: process.env.EXE });
  const p = await b.newPage({ viewport: { width: 1600, height: 1700 } });
  await p.goto('file://' + __dirname + '/make.html');
  await p.waitForSelector('body[data-ready="1"]');
  await (await p.$('#logo')).screenshot({ path: __dirname + '/logo-1024.png' });
  await (await p.$('#banner')).screenshot({ path: __dirname + '/x-banner-1500x500.png' });
  await b.close();
})();
