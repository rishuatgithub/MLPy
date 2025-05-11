import os
import uuid
import boto3
from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import ClientError


class S3Utils:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.bucket = boto3.resource("s3").Bucket(self.bucket_name)

    def create(self):
        """
        Create an S3 bucket
        :return: bucket_name
        """
        try:
            self.bucket.create(
                CreateBucketConfiguration={"LocationConstraint": self.bucket.meta.client.meta.region_name})
            print(f"Bucket Created. Bucket Name: {bucket_name}")
            return bucket_name
        except ClientError as err:
            print(f"Tried and failed to create demo bucket {bucket_name}.")
            print(f"\t{err.response['Error']['Code']}:{err.response['Error']['Message']}")
            print(f"\nCan't continue the demo without a bucket!")
            return

    def upload_files(self, filename):
        """
        Upload the files to S3
        :param filename: name of the file to upload
        :return:
        """
        file_to_upload = os.path.basename(f"data/{filename}")
        obj = self.bucket.Object(file_to_upload)
        try:
            obj.upload_file(filename)
            print(f"Uploaded file: {filename} to bucket: {bucket_name}")
        except S3UploadFailedError as err:
            print(f"Couldn't upload the file {filename} to {bucket_name}")
            print(f"\t{err}")

    def delete(self):
        """
        delete an s3 bucket
        :return:
        """
        print("\nYour bucket contains the following objects:")
        try:
            for o in self.bucket.objects.all():
                print(f"\t{o.key}")
        except ClientError as err:
            print(f"Couldn't list the objects in bucket {self.bucket.name}.")
            print(f"\t{err.response['Error']['Code']}:{err.response['Error']['Message']}")

        answer = input("\nDo you want to delete all of the objects as well as the bucket (y/n)? ")
        if answer.lower() == "y":
            try:
                self.bucket.objects.delete()
                self.bucket.delete()
                self.bucket.wait_until_not_exists()
                print(f"Emptied and deleted bucket {self.bucket.name}.\n")
            except ClientError as err:
                print(f"Couldn't empty and delete bucket {self.bucket.name}.")
                print(f"\t{err.response['Error']['Code']}:{err.response['Error']['Message']}")


if __name__ == '__main__':
    bucket_name = f"aws-ml-cert-bucket-{uuid.uuid4()}"
    file_name = os.path.abspath("../materials/AWSMachineLearning/book.txt")
    s3 = S3Utils(bucket_name)

    s3.create()
    s3.upload_files(file_name)
    s3.delete()
