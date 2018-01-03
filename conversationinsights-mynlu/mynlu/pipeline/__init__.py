from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging
import os
from collections import defaultdict
import importlib
import pkg_resources
import typing
from builtins import object
import inspect

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Text
from typing import Tuple

logger = logging.getLogger(__name__)


def validate_requirements(plugins, dev_requirements_file="dev-requirements.txt"):
    # type: (List[Text], Text) -> None
    """Ensures that all required python packages are installed to instantiate and used the passed plugins."""
    from mynlu import registry

    failed_imports = set()
    for plugin_name in plugins:
        plugin = registry.get_plugin(plugin_name)
        failed_imports.update(_find_unavailable_packages(plugin.required_packages()))
    if failed_imports:
        # if available, use the development file to figure out the correct version numbers for each requirement
        all_requirements = _read_dev_requirements(dev_requirements_file)
        if all_requirements:
            missing_requirements = [r for i in failed_imports for r in all_requirements[i]]
            raise Exception("Not all required packages are installed. " +
                            "Failed to find the following imports {}. ".format(", ".join(failed_imports)) +
                            "To use this pipeline, you need to install the missing dependencies, including:\n\t" +
                            "{}".format(" ".join(missing_requirements)))
        else:
            raise Exception("Not all required packages are installed. " +
                            "To use this pipeline, you need to install the missing dependencies. " +
                            "Please install {}".format(", ".join(failed_imports)))


def _find_unavailable_packages(package_names):
    # type: (List[Text]) -> Set[Text]

    failed_imports = set()
    for package in package_names:
        try:
            importlib.import_module(package)
        except ImportError:
            failed_imports.add(package)
    return failed_imports


def _read_dev_requirements(file_name):
    try:
        req_lines = pkg_resources.resource_string("mynlu", "../" + file_name).split("\n")
    except Exception as e:
        logger.info("Couldn't read dev-requirements.txt. Error: {}".format(e))
        req_lines = []
    return _requirements_from_lines(req_lines)


def _requirements_from_lines(req_lines):
    requirements = defaultdict(list)
    current_name = None
    for req_line in req_lines:
        if req_line.startswith("#"):
            current_name = req_line[1:].strip(' \n')
        elif current_name is not None:
            requirements[current_name].append(req_line.strip(' \n'))
    return requirements


def validate_arguments(pipeline, context, allow_empty_pipeline=False):
    # type: (List[Component], Dict[Text, Any], bool) -> None
    """Validates a pipeline before it is run. Ensures, that all arguments are present to train the pipeline."""

    # Ensure the pipeline is not empty
    if not allow_empty_pipeline and len(pipeline) == 0:
        raise ValueError("Can not train an empty pipeline. " +
                         "Make sure to specify a proper pipeline in the configuration using the `pipeline` key." +
                         "The `backend` configuration key is NOT supported anymore.")

    provided_properties = set(context.keys())

    for plugin in pipeline:
        for req in plugin.requires:
            if req not in provided_properties:
                raise Exception("Failed to validate at plugin '{}'. Missing property: '{}'".format(
                    plugin.name, req))
        provided_properties.update(plugin.provides)


class MissingArgumentError(ValueError):
    """Raised when a function is called and not all parameters can be filled from the context / config.

    Attributes:
        message -- explanation of which parameter is missing
    """

    def __init__(self, message):
        # type: (Text) -> None
        super(MissingArgumentError, self).__init__(message)
        self.message = message

    def __str__(self):
        return self.message
