# Inverse Kinematics using classical Jacobian

This program simulates a Cartesian robot which has 3 links and can only move on planer surface.

It has 3 joint angles (Q1,Q2,Q3) with constrains for each joint angle.
Q1 can only go from 0 to (-180) degrees.
Q2 can only go from (-180) to 0 degrees.
Q3 can only go from (-90) to 90 degrees.

The robot has 3 links: A, B and C.
Link A starts in Q1 and ends in Q2.
Link B starts in Q2 and ends in Q3.
Link C starts in Q3 and ends in the robot end effector or just robot end.

It uses de classical Jacobian. Plese see this blog:
https://medium.com/unity3danimation/overview-of-jacobian-ik-a33939639ab2

