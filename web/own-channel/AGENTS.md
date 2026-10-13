# Local-only development

The user requested local-only operation on 2026-10-12. Keep both model tabs
available through the local preview server. Do not push source to Sites,
register a new Site, package for cloud publication, or deploy updates unless
the user explicitly asks to publish again.

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
