# Local preview and GitHub Pages

The user authorized a public GitHub repository and GitHub Pages publication on
2026-10-15. The Pages workflow builds and deploys the static browser lab from
`master`. Keep the local preview available. Settings must use the five prebuilt
widths 16, 32, 64, 128 and 256, with custom learning rate and weight decay.

Do not push source to Sites, register a new Site, or deploy there unless the
user explicitly asks to use Sites again.

The existing `.openai/hosting.json` retains the old Site identity for removal
or later recovery. Its presence does not authorize publication. The currently
available Sites tools do not provide deletion or unpublication; do not describe
the cloud copy as removed without a confirmed removal result.

Run the local server with Node selected by `.nvmrc`:

```sh
cd web/own-channel
npm run build
npm start
```

The preview binds to `127.0.0.1:8080`. Training runs in the browser.
