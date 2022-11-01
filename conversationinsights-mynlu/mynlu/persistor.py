from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import object
import os
import shutil
import tarfile
import io

import boto3
import botocore
from typing import Optional
from typing import Text
from mynlu.config.mynluconfig import MyNLUConfig


def get_persistor(config):
    # type: (MyNLUConfig) -> Optional[Persistor]
    """Returns an instance of the requested persistor. Currently, `aws` and `gcs` are supported"""

    if 'storage' not in config:
        raise KeyError("No persistent storage specified. Supported values are {}".format(", ".join(['aws', 'gcs'])))

    if config['storage'] == 'cloud':
        return CloudPersistor(config['path'], config['aws_region'], config['bucket_name'], config['aws_endpoint_url'])
    else:
        return None


class Persistor(object):
    """Store models in cloud and fetch them when needed"""

    def save_tar(self, target_dir):
        # type: (Text) -> None
        """Uploads a model persisted in the `target_dir` to cloud storage."""
        raise NotImplementedError("")

    def fetch_and_extract(self, filename):
        # type: (Text) -> None
        """Downloads a model that has previously been persisted to cloud storage."""
        raise NotImplementedError("")

class CloudPersistor(Persistor):
    """Store models on cloud and fetch them when needed instead of storing them on the local disk."""

    def __init__(self, data_dir, aws_region, bucket_name, endpoint_url):
        # type: (Text, Text, Text) -> None
        Persistor.__init__(self)
        self.data_dir = data_dir
        self.s3 = boto3.resource('s3', region_name=aws_region, endpoint_url=endpoint_url)
        self.bucket_name = bucket_name
        try:
            self.s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': aws_region})
        except botocore.exceptions.ClientError:
            pass  # bucket already exists
        self.bucket = self.s3.Bucket(bucket_name)

    def save_tar(self, target_dir):
        # type: (Text) -> None
        """Uploads a model persisted in the `target_dir` to cloud."""

        if not os.path.isdir(target_dir):
            raise ValueError("Target directory '{}' not found.".format(target_dir))

        base_name = os.path.basename(target_dir)
        base_dir = os.path.dirname(target_dir)
        tarname = shutil.make_archive(base_name, 'gztar', root_dir=base_dir, base_dir=base_name)
        filekey = os.path.basename(tarname)
        self.s3.Object(self.bucket_name, filekey).put(Body=open(tarname, 'rb'))

    def fetch_and_extract(self, filename):
        # type: (Text) -> None
        """Downloads a model that has previously been persisted to s3."""

        with io.open(filename, 'wb') as f:
            self.bucket.download_fileobj(filename, f)
        with tarfile.open(filename, "r:gz") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, self.data_dir)
