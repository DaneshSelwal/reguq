# Security policy

reguq is a numerical library. It does not open network connections or deserialize untrusted pickles.
`PipelineConfig.from_file` uses `yaml.safe_load`.

Report a vulnerability privately through GitHub's *Report a vulnerability* (Security tab), not in a
public issue. Expect an acknowledgement within 7 days.

Supported versions: the latest minor release.
