# Privacy

## Data Collection
The software may collect information about you and your use of the software and send it to Microsoft. Microsoft may use this information to provide services and improve our products and services. You may turn off the telemetry as described in the repository. There are also some features in the software that may enable Microsoft to collect data from users of your applications. If you use these features, you must comply with applicable law, including providing appropriate notices to users of your applications together with a copy of Microsoft's privacy statement. Our privacy statement can be found [here](https://go.microsoft.com/fwlink/?LinkID=824704). You can learn more about data collection and use in the help documentation and our privacy statement. Your use of the software operates as your consent to these practices.

***

## Technical Details
Telemetry is turned ON by default. Based on user consent, this data may be periodically sent to Microsoft servers following GDPR and privacy regulations for anonymity and data access controls. Application, device, and version information is collected automatically.

In addition, Olive may collect additional telemetry data such as:
- Invoked commands
- Performance data
- Exception information

You can fully disable telemetry by adding the `--disable_telemetry` flag to any Olive CLI command, setting `OLIVE_DISABLE_TELEMETRY=1` or `ORT_DISABLE_TELEMETRY=1` before running, or calling `olive.telemetry.disable_telemetry()`. Each option suppresses every subsequent Olive telemetry event for the remainder of the process, including Olive workflow containers started by that process. When the opt-out is active before first telemetry use, Olive does not construct the telemetry singleton or create the telemetry queue, uploader, or persistent device identifier. Disabling at runtime stops this process's uploader, retains already queued unsent rows unchanged for a later telemetry-enabled process, and does not enqueue another Heartbeat. The environment variables accept `1`, `true`, `yes`, `on`, or `y` after trimming and without regard to case.

In CI/CD environments (e.g., GitHub Actions, Azure Pipelines, Jenkins), Olive suppresses the device-id heartbeat and the action/error events and only emits the `OliveRecipe` event. Any full opt-out takes precedence and sends nothing. The `OliveRecipe` event may include recipe metadata such as pass types, explicitly configured target settings, the host system type (including the default `LocalSystem` host) and any explicitly configured host accelerator settings, whether a custom package config was provided, a redacted snapshot of custom package-config overrides, and a redacted snapshot of explicitly supplied config overrides.

Telemetry is implemented using only the Python standard library. In enabled local runs, one `OliveHeartbeat` per process and detailed events are written to a local per-user SQLite queue before a background uploader sends them to Microsoft over HTTPS. Olive first reserves a minimal Heartbeat durably, then adds available operating-system metadata before making it eligible for upload; if enrichment is interrupted, the minimal Heartbeat remains eligible for a later delivery attempt. CI recipe events use a separate recipe-only queue and receive a bounded shutdown delivery attempt. Events that cannot be sent remain in the applicable queue for a later run. The transmitted `deviceId` is the `c:`-prefixed SHA-256 hash of a shared persistent UUID; the raw UUID is not transmitted.

### Event schemas

All events include Olive-assigned `appName`, `LibraryVersion`, and `AppSessionGuid` values that event callers cannot override; `initTs` is included when supplied by the caller. Olive emits only the following event-specific fields:

| Event | Fields |
| --- | --- |
| `OliveHeartbeat` | `deviceId`, `deviceIdStatus`; `os`, `osVersion`, `osRelease`, and `osArchitecture` when enrichment succeeds |
| `OliveAction` | `invokedFrom`, `actionName`, `durationMs`, `success` |
| `OliveError` | `exceptionType`, `exceptionMessage` |
| `OliveRecipe` | `recipeName`, `recipeHash`, `recipeSource`, `recipeFormat`, `recipeCommand`, `executionMode`, `workflowId`, `configOverrides`, `success`, `inputModelType`, `inputModelSource`, `modelTask`, `targetSystemType`, `targetDevice`, `targetExecutionProvider`, `targetExecutionProviders`, `hostSystemType`, `hostDevice`, `hostExecutionProvider`, `hostExecutionProviders`, `passTypes`, `passCount`, `dataConfigCount`, `searchEnabled`, `packageConfigProvided`, `packageConfigOverrides`, `isCI` |

Free-text values, paths, URLs, query secrets, credential-bearing configuration keys, environment-variable values, and nested configuration metadata are recursively redacted at the serialization boundary and capped at 40,960 UTF-8 bytes. `recipeHash` is computed only after credential, environment-value, and path redaction. Error messages may contain sanitized exception and frame metadata but never source-code lines.
