import boto3
from base64 import b64encode


if __name__ == '__main__':		
	ec2 = boto3.client('ec2')
	
	for i in range(INST_COUNT):
		pre_USER_DATA = '''#!/bin/bash
		python /home/ec2-user/download_mean_shift.py
		python /home/ec2-user/mean_shift.py ''' + str(i) + ' ' + str(INST_COUNT)
		
		USER_DATA = b64encode(pre_USER_DATA)		
		response = ec2.request_spot_instances(
			SpotPrice= '.015',
			InstanceCount=1,
			Type='one-time',
			LaunchSpecification={
				'ImageId': AMI,
				'InstanceType': 'm4.large',
				'KeyName' : KEY,
				'UserData': USER_DATA,
				# 'SecurityGroups' : [SG],
				'SecurityGroupIds' : [SG_ID],
				'IamInstanceProfile' : {
					'Arn' : S3_ARN
				}
			}
		)