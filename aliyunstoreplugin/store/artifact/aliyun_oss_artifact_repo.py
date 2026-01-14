import os
import posixpath
import oss2
from six.moves import urllib

# 阿里云相关依赖
from alibabacloud_credentials.client import Client
from oss2 import CredentialsProvider
from oss2.credentials import Credentials

# MLflow 依赖
from mlflow.entities import FileInfo
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils.file_utils import relative_path_to_artifact_path


class CredentialProviderWrapper(CredentialsProvider):
    def __init__(self, client):
        self.client = client

    def get_credentials(self):
        access_key_id = self.client.get_access_key_id()
        access_key_secret = self.client.get_access_key_secret()
        security_token = self.client.get_security_token()
        return Credentials(access_key_id, access_key_secret, security_token)


class AliyunOssArtifactRepository(ArtifactRepository):
    """Stores artifacts on Aliyun OSS."""

    # ------------------------------------------------------------------------
    # [兼容性修复] MLflow 3.0+ 在初始化 Repository 时会传入 experiment_id 和 run_id
    # 必须修改 __init__ 接收这些参数，否则会报错。
    # 建议加上 **kwargs 以防止未来 MLflow 增加参数导致再次不兼容。
    # ------------------------------------------------------------------------
    def __init__(self, artifact_uri, experiment_id=None, run_id=None, oss_bucket=None, **kwargs):
        # 调用父类初始化
        super(AliyunOssArtifactRepository, self).__init__(artifact_uri)
        
        # 保存 experiment_id 和 run_id (虽然 OSS 上传逻辑可能暂时不用，但最好存下来)
        self.experiment_id = experiment_id
        self.run_id = run_id

        if oss_bucket is not None:
            self.oss_bucket = oss_bucket
            return

        self.auth = None
        self.oss_endpoint_url = os.environ.get("MLFLOW_OSS_ENDPOINT_URL")
        
        # 建议：使用异常替代 assert，对用户更友好
        if not self.oss_endpoint_url:
            raise MlflowException("Please set MLFLOW_OSS_ENDPOINT_URL environment variable.")

        oss_access_key_id = os.environ.get("MLFLOW_OSS_KEY_ID")
        oss_access_key_secret = os.environ.get("MLFLOW_OSS_KEY_SECRET")
        
        if oss_access_key_id and oss_access_key_secret:
            self.auth = oss2.Auth(oss_access_key_id, oss_access_key_secret)
        else:
            try:
                cred = Client()
                credentials_provider = CredentialProviderWrapper(cred)
                self.auth = oss2.ProviderAuth(credentials_provider)
            except Exception:
                pass
        
        if not self.auth:
            raise MlflowException(
                "Please set MLFLOW_OSS_KEY_ID and MLFLOW_OSS_KEY_SECRET environment variable, "
                "or refer to alibabacloud-credentials for default provider configuration."
            )

        self.oss_bucket = None
        # is_plugin 标记在旧版中有用，保留以防万一
        self.is_plugin = True

    @staticmethod
    def parse_oss_uri(uri):
        """Parse an OSS URI, returning (bucket, path)"""
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme != "oss":
            raise Exception("Not an OSS URI: %s" % uri)
        path = parsed.path
        if path.startswith("/"):
            path = path[1:]
        return parsed.netloc, path

    def _get_oss_bucket(self, bucket):
        if self.oss_bucket is not None:
            return self.oss_bucket
        # 注意：oss2.Bucket 初始化非常快（只是个客户端对象），不涉及网络请求
        self.oss_bucket = oss2.Bucket(self.auth, self.oss_endpoint_url, bucket)
        return self.oss_bucket

    def log_artifact(self, local_file, artifact_path=None):
        (bucket, dest_path) = self.parse_oss_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        dest_path = posixpath.join(dest_path, os.path.basename(local_file))
        self._get_oss_bucket(bucket)
        self.oss_bucket.put_object_from_file(dest_path, local_file)

    def log_artifacts(self, local_dir, artifact_path=None):
        (bucket, dest_path) = self.parse_oss_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        self._get_oss_bucket(bucket)
        local_dir = os.path.abspath(local_dir)
        for root, _, filenames in os.walk(local_dir):
            upload_path = dest_path
            if root != local_dir:
                rel_path = os.path.relpath(root, local_dir)
                rel_path = relative_path_to_artifact_path(rel_path)
                upload_path = posixpath.join(dest_path, rel_path)
            for f in filenames:
                self.oss_bucket.put_object_from_file(
                    posixpath.join(upload_path, f), os.path.join(root, f)
                )

    def list_artifacts(self, path=None):
        (bucket, artifact_path) = self.parse_oss_uri(self.artifact_uri)
        dest_path = artifact_path
        if path:
            dest_path = posixpath.join(dest_path, path)
        infos = []
        prefix = dest_path + "/" if dest_path else ""
        self._get_oss_bucket(bucket)
        
        # 注意：如果文件特别多，这里可能需要处理分页 (IsTruncated)，原版插件似乎没处理分页
        results = self.oss_bucket.list_objects(prefix=prefix, delimiter="/")

        for obj in results.object_list:
            # is file
            file_path = obj.key
            self._verify_listed_object_contains_artifact_path_prefix(
                listed_object_path=file_path, artifact_path=artifact_path
            )
            file_rel_path = posixpath.relpath(path=file_path, start=artifact_path)
            file_size = obj.size
            infos.append(FileInfo(file_rel_path, False, file_size))

        for subdir_path in results.prefix_list:
            # is dir
            self._verify_listed_object_contains_artifact_path_prefix(
                listed_object_path=subdir_path, artifact_path=artifact_path
            )
            subdir_rel_path = posixpath.relpath(path=subdir_path, start=artifact_path)
            infos.append(FileInfo(subdir_rel_path, True, None))
        return sorted(infos, key=lambda f: f.path)

    @staticmethod
    def _verify_listed_object_contains_artifact_path_prefix(
        listed_object_path, artifact_path
    ):
        if not listed_object_path.startswith(artifact_path):
            raise MlflowException(
                "The path of the listed oss object does not begin with the specified"
                " artifact path. Artifact path: {artifact_path}. Object path:"
                " {object_path}.".format(
                    artifact_path=artifact_path, object_path=listed_object_path
                )
            )

    def _download_file(self, remote_file_path, local_path):
        (bucket, oss_root_path) = self.parse_oss_uri(self.artifact_uri)
        oss_full_path = posixpath.join(oss_root_path, remote_file_path)
        self._get_oss_bucket(bucket)
        self.oss_bucket.get_object_to_file(oss_full_path, local_path)

    def delete_artifacts(self, artifact_path=None):
        # MLflow 3.0 的 GC 机制可能会调用此方法，如果不实现，删除 Run 时可能无法清理 OSS 文件
        # 如果暂时不实现，保留 raise 也可以，只是后台会报错
        raise MlflowException("Delete artifacts not implemented yet for Aliyun OSS")