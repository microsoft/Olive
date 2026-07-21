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

You can disable telemetry by adding the `--disable_telemetry` flag to any Olive CLI command, or by setting the `ORT_DISABLE_TELEMETRY` environment variable to `1` before running. When telemetry is disabled this way, the additional telemetry above (commands, performance, exceptions) is not sent. A minimal device-id heartbeat — a non-reversible hashed device identifier plus basic operating-system name, version, release, and architecture — is still sent outside CI/CD environments so Microsoft can count active devices; it contains no command, performance, or exception data.

In CI/CD environments (e.g., GitHub Actions, Azure Pipelines, Jenkins), Olive suppresses the device-id heartbeat and the action/error events and only emits the `OliveRecipe` event. The `OliveRecipe` event may include recipe metadata such as pass types, explicitly configured target settings, the host system type (including the default `LocalSystem` host) and any explicitly configured host accelerator settings, whether a custom package config was provided, a redacted snapshot of custom package-config overrides, and a redacted snapshot of explicitly supplied config overrides. Setting `ORT_DISABLE_TELEMETRY=1` in a CI/CD environment sends nothing at all.

Telemetry is implemented using only the Python standard library. Events are written to a local per-user SQLite queue and uploaded in the background to Microsoft over HTTPS. If telemetry is enabled but cannot be sent (for example, while offline), events remain in the local queue and are uploaded on a later run when a connection is available.
