from collections import defaultdict, OrderedDict

from onnx import defs


class OnnxOpSchema(object):
    def __init__(self, name, domain, since_version, attributes):
        self.name = name
        self.domain = domain
        self.attributes = attributes
        self.since_version = since_version

    @staticmethod
    def from_onnx_schema(onnx_schema):
        name = onnx_schema.name
        domain = onnx_schema.domain
        since_version = onnx_schema.since_version
        attributes = onnx_schema.attributes
        return OnnxOpSchema(name, domain, since_version, attributes)


def _register_all_schemas_with_history():
    """Register all schemas with history"""
    onnx_schemas = defs.get_all_schemas_with_history()
    namedomain_version_schema_map = defaultdict(lambda: defaultdict(dict))
    for s in onnx_schemas:
        schema = OnnxOpSchema(s.name, s.domain, s.since_version, s.attributes)
        namedomain_version_schema_map[schema.name][schema.domain][schema.since_version] = schema

    ordered_map = defaultdict(lambda: defaultdict(OrderedDict))
    for name, domain_version_schema_map in namedomain_version_schema_map.items():
        for domain, version_schema_map in domain_version_schema_map.items():
            ordered_map[name][domain] = OrderedDict(
                sorted(version_schema_map.items(), key=lambda x: -x[0]))

    return ordered_map


def _parse_domain_opset_versions(schemas):
    """ Get max opset version among all schemas within each domain. """
    domain_opset_versions = dict()
    for domain_version_schema_map in schemas.values():
        for domain, version_schema_map in domain_version_schema_map.items():
            # version_schema_map is sorted by since_version in descend order
            max_version = next(iter(version_schema_map))
            if domain not in domain_opset_versions:
                domain_opset_versions[domain] = int(max_version)
            else:
                domain_opset_versions[domain] = max(domain_opset_versions[domain], int(max_version))
    return domain_opset_versions


def get_max_opset_version():
    _schemas = _register_all_schemas_with_history()
    versions = _parse_domain_opset_versions(_schemas)
    return versions.get("")
