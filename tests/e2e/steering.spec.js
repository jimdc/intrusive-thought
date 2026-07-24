import { test, expect } from "@playwright/test";

const VIEWPORTS = [
  { name: "mobile-390", width: 390, height: 844 },
  { name: "desktop-1440", width: 1440, height: 900 },
];

for (const viewport of VIEWPORTS) {
  test.describe(`${viewport.name}`, () => {
    test.use({ viewport: { width: viewport.width, height: viewport.height } });

    test(`loads both banks with no horizontal overflow (${viewport.name})`, async ({ page }) => {
      await page.goto("/");
      await expect(page.locator("#demo")).toBeVisible();

      const saeChips = page.locator("#chipsSae .chip");
      const actaddChips = page.locator("#chipsActadd .chip");
      await expect(saeChips.first()).toBeVisible();
      await expect(actaddChips.first()).toBeVisible();
      expect(await saeChips.count()).toBeGreaterThan(0);
      expect(await actaddChips.count()).toBeGreaterThan(0);

      const overflow = await page.evaluate(() => {
        const doc = document.documentElement;
        return doc.scrollWidth - doc.clientWidth;
      });
      expect(overflow).toBeLessThanOrEqual(1);

      await page.screenshot({
        path: `tests/e2e/screenshots/${viewport.name}-default.png`,
        fullPage: true,
      });
    });

    test(`ActAdd chip surfaces the contrast pair (${viewport.name})`, async ({ page }) => {
      await page.goto("/");
      await expect(page.locator("#demo")).toBeVisible();

      const firstActadd = page.locator("#chipsActadd .chip").first();
      await firstActadd.click();
      await expect(firstActadd).toHaveClass(/active/);

      const contrastPair = page.locator(".contrast-pair");
      await expect(contrastPair).toBeVisible();
      await expect(page.locator(".contrast-side.pos .contrast-text")).not.toBeEmpty();
      await expect(page.locator(".contrast-side.neg .contrast-text")).not.toBeEmpty();

      const overflow = await page.evaluate(() => {
        const doc = document.documentElement;
        return doc.scrollWidth - doc.clientWidth;
      });
      expect(overflow).toBeLessThanOrEqual(1);

      await page.screenshot({
        path: `tests/e2e/screenshots/${viewport.name}-actadd.png`,
        fullPage: true,
      });
    });

    test(`slider changes the steered output and returns to baseline at 0 (${viewport.name})`, async ({ page }) => {
      await page.goto("/");
      await expect(page.locator("#demo")).toBeVisible();

      const slider = page.locator("#strength");
      const steeredOut = page.locator("#steeredOut");
      const baselineText = await steeredOut.innerText();
      const zeroAt = await slider.inputValue(); // selectTour() parks the slider at zeroAt on load

      const max = Number(await slider.getAttribute("max"));
      await slider.fill(String(max));
      await expect(async () => {
        expect(await steeredOut.innerText()).not.toBe(baselineText);
      }).toPass();

      await slider.fill(zeroAt);
      await expect(page.locator("#strengthVal")).toHaveText("off");
      await expect(steeredOut).toHaveText(await page.locator("#defaultOut").innerText());
    });

    test(`SAE feature bank still renders its provenance line (${viewport.name})`, async ({ page }) => {
      await page.goto("/");
      await expect(page.locator("#demo")).toBeVisible();

      const firstSae = page.locator("#chipsSae .chip").first();
      await firstSae.click();
      await expect(firstSae).toHaveClass(/active/);
      await expect(page.locator("#feat")).toContainText("the model’s own feature for");

      const overflow = await page.evaluate(() => {
        const doc = document.documentElement;
        return doc.scrollWidth - doc.clientWidth;
      });
      expect(overflow).toBeLessThanOrEqual(1);
    });
  });
}
