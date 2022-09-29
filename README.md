# pyscallop
A computer vision approach to the chalenge of sorting scallops fast and reliably for the highly competitive and strictly regulated harvesting in France.

## Why?
Scallop harvesting is extremely regulated. One cannot harvest for more than 45 minutes per session, and it is illegal to keep scallops under $10.2cm$ on the boat after those 45 minutes. Only three sailers are allowd per boat and one of them is completely occupied by the daunting task of sorting all those scallops.

The software is used to detect and measure the scallops it's given.

![[scal.png]]

## How?
With a simple converyor belt and a cheap camera (and some good lightning), and some computer vision code writen on OpenCV for Python. The recognission is solely based on the unique tint of the scallops.

## Is it precise?
According to the needs of my project, the software is precise enough:
![[precision.png]]
