# Privacy

## Data Collection
The software may collect information about you and your use of the software and send it to Microsoft. Microsoft may use this information to provide services and improve our products and services. You may turn off the telemetry as described in the repository. There are also some features in the software that may enable Microsoft to collect data from users of your applications. If you use these features, you must comply with applicable law, including providing appropriate notices to users of your applications together with a copy of Microsoft's privacy statement. Our privacy statement can be found [here](https://go.microsoft.com/fwlink/?LinkID=824704). You can learn more about data collection and use in the help documentation and our privacy statement. Your use of the software operates as your consent to these practices.

***

## Telemetry
Telemetry is turned ON by default. It may include workflow and build metadata, device and platform information, performance measurements, diagnostics, and a pseudonymous device identifier. Local paths and credentials are redacted.

## Disabling Telemetry

- Add `--disable_telemetry` to an Olive CLI command.
- Set `OLIVE_DISABLE_TELEMETRY=1` or `ORT_DISABLE_TELEMETRY=1` before running.
- Call `olive.telemetry.disable_telemetry()`.

Opt-out applies for the rest of the process and to Olive workflow containers it starts. Runtime opt-out stops new events and uploads. An upload already admitted may finish, and retained queued events may be sent by a later telemetry-enabled process. Opt out before first use to prevent all uploads.

In CI, Olive sends only recipe metadata unless telemetry is disabled.
