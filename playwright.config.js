import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: true,
  reporter: [["list"]],
  use: {
    baseURL: "http://127.0.0.1:8123",
  },
  webServer: {
    command: "python3 -m http.server 8123",
    url: "http://127.0.0.1:8123",
    reuseExistingServer: !process.env.CI,
  },
});
