import boto3


class S3():
    def __init__(self):
        self.s3 = boto3.resource('s3')

    def create_bucket(self, bucket_name):
        self.s3.create_bucket(Bucket=bucket_name)
        # versioning = self.s3.BucketVersioning(bucket_name)
        # versioning.enable()

    def create_object(self, bucket_name, object_key, content=None):
        object = self.s3.Object(bucket_name, object_key)
        if content != None:
            object.put(Body=content)

    # def update_object(self, bucket_name, object_key, content):
    #     self.delete_object(bucket_name, object_key)
    #     self.create_object(bucket_name, object_key, content)

    def get_object(self, bucket_name, object_key):
        object = self.s3.Object(bucket_name, object_key)
        object_content = object.get()['Body'].read().decode()
        return object_content

    def list_all(self):
        for bucket in self.s3.buckets.all():
            print('\nbucket: {}'.format(bucket.name))
            for obj in bucket.objects.all():
                print('object: {}'.format(obj.key))
                # versions = self.s3.Bucket(bucket.name).object_versions.filter(Prefix=obj.key)
                # for version in versions:
                #     obj_ver = version.get()
                #     print(obj_ver.get('VersionId'), obj_ver.get('Body').read().decode(), obj_ver.get('LastModified'))

    def delete_object(self, bucket_name, object_key):
        object = self.s3.Object(bucket_name, object_key)
        object.delete()
        # versions = self.s3.Bucket(bucket_name).object_versions.filter(Prefix=object_key)
        # for version in versions:
        #     obj_ver = version.get()
        #     object.delete(VersionId=obj_ver.get('VersionId'))

    def delete_file(self, bucket_name, file_name):
        bucket = self.s3.Bucket(bucket_name)
        for obj in bucket.objects.all():
            if (obj.key.startswith(file_name)):
                obj.delete()

    def delete_bucket(self, bucket_name):
        # All of the keys in a bucket must be deleted
        # before the bucket itself can be deleted

        bucket = self.s3.Bucket(bucket_name)
        for obj in bucket.objects.all():
            obj.delete()
            # versions = self.s3.Bucket(bucket.name).object_versions.filter(Prefix=obj.key)
            # for version in versions:
            #     obj_ver = version.get()
            #     obj.delete(VersionId=obj_ver.get('VersionId'))

        bucket.delete()

    def delete_all(self):
        for bucket in self.s3.buckets.all():
            self.delete_bucket(bucket.name)


if __name__ == '__main__':
    client = S3()
    bucket_name = 'pinyugly'
    file_name = '1/'
    # client.create_bucket(bucket_name)
    # client.create_object(bucket_name, obj_key)
    # client.create_object(bucket_name, obj_key+'content', 'what<br>th')
    # client.update_object(bucket_name, obj_key+'content', 'what<br>thopo')
    # object_content = client.get_object(bucket_name, obj_key+'content')
    # print('\n---------')
    # print(object_content)
    # object_content = client.get_object(bucket_name, '2/content')
    # print('\n---------')
    # print(object_content)
    # client.delete_object(bucket_name, obj_key+'content')
    # client.delete_bucket(bucket_name)
    # client.delete_object(bucket_name, obj_key)
    # client.delete_file(bucket_name, file_name)
    # client.delete_all()
    client.create_bucket(bucket_name)
    client.list_all()