# Privacy

## Data Collection
The software may collect information about you and your use of the software and send it to Microsoft. Microsoft may use this information to provide services and improve our products and services. You may turn off the telemetry as described in the repository. There are also some features in the software that may enable Microsoft to collect data from users of your applications. If you use these features, you must comply with applicable law, including providing appropriate notices to users of your applications together with a copy of Microsoft's privacy statement. Our privacy statement can be found [here](https://go.microsoft.com/fwlink/?LinkID=824704). You can learn more about data collection and use in the help documentation and our privacy statement. Your use of the software operates as your consent to these practices.

***

## Technical Details
Telemetry is turned ON by default. Based on user consent, this data may be periodically sent to Microsoft servers following GDPR and privacy regulations for anonymity and data access controls.

You can fully disable telemetry by adding the `--disable_telemetry` flag to any Olive CLI command, setting `OLIVE_DISABLE_TELEMETRY=1` or `ORT_DISABLE_TELEMETRY=1` before running, or calling `olive.telemetry.disable_telemetry()`. Each option suppresses every subsequent Olive telemetry event for the remainder of the process, including Olive workflow containers started by that process. When the opt-out is active before first telemetry use, Olive does not construct the telemetry singleton or create the telemetry queue, uploader, or persistent device identifier. Disabling during runtime stops new events and upload admissions without waiting for transport work, and retains already queued unsent rows unchanged for a later telemetry-enabled process. An upload already admitted to the transport may finish after the call returns; runtime opt-out does not revoke an admitted request. To prevent all uploads, opt out before first telemetry use. The environment variables accept `1`, `true`, `yes`, `on`, or `y` after trimming and without regard to case.

In CI/CD environments (e.g., GitHub Actions, Azure Pipelines, Jenkins), Olive only emits the `OliveRecipe` event with recipe metadata. Any full opt-out takes precedence and sends nothing.

File-based telemetry storage rejects symlinked or reparse-point paths and foreign-owned storage before persisting identifiers or events. POSIX uses directory-relative no-follow opens and private permissions; Windows pins validated paths with handles that prevent replacement while in use. Unsafe queue storage disables the queue and uploader. Storage and device-ID failures are available through telemetry debug logging; an unavailable persistent identifier falls back to one process-cached identifier.

On macOS, the root-owned system aliases `/etc`, `/tmp`, and `/var` are accepted only when they point to their expected `/private` directories and their parent is protected from non-root writes. All remaining path components retain the no-follow checks; arbitrary user-created symlinks are still rejected.
