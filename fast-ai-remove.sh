#!/bin/bash
aws ec2 disassociate-address --association-id eipassoc-493ae931
aws ec2 release-address --allocation-id eipalloc-8ae3eaee
aws ec2 terminate-instances --instance-ids i-08d904114000f4890
aws ec2 wait instance-terminated --instance-ids i-08d904114000f4890
aws ec2 delete-security-group --group-id sg-99d5f6ff
aws ec2 disassociate-route-table --association-id rtbassoc-518ab436
aws ec2 delete-route-table --route-table-id rtb-d09c5db7
aws ec2 detach-internet-gateway --internet-gateway-id igw-a93697cd --vpc-id vpc-e47a5280
aws ec2 delete-internet-gateway --internet-gateway-id igw-a93697cd
aws ec2 delete-subnet --subnet-id subnet-afd0edd9
aws ec2 delete-vpc --vpc-id vpc-e47a5280
echo If you want to delete the key-pair, please do it manually.
