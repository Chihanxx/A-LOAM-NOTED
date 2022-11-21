// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
/*
该节点主要用到来自cornerPointsLessSharp和surfPointsLessFlatScan的数据，对这两个容器中的点云进行了降采样，基于PCA原理，使用ceres求解器计算出两帧之间的位姿。
在函数laserOdometryHandler中，将laserOdometry节点和laserMapping节点计算的位姿结合，即可得到最终的轨迹odomAftMapped

该文件的功能：基于前述的corner_less_sharp特征点和surf_less_flat特征点，进行帧与submap的点云特征配准，对Odometry线程计算的位姿进行finetune
*/
#include <math.h>
#include <vector>
#include <aloam_velodyne/common.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>

#include "lidarFactor.hpp"
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"


int frameCount = 0;

double timeLaserCloudCornerLast = 0;
double timeLaserCloudSurfLast = 0;
double timeLaserCloudFullRes = 0;
double timeLaserOdometry = 0;


// 维护这些CUBE来获得局部地图的
int laserCloudCenWidth = 10;
int laserCloudCenHeight = 10;
int laserCloudCenDepth = 5;
const int laserCloudWidth = 21;
const int laserCloudHeight = 21;
const int laserCloudDepth = 11;

//cube的数量
const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth; //4851


// 记录submap中的有效cube的index，注意submap中cube的最大数量为 5 * 5 * 5 = 125
int laserCloudValidInd[125];
int laserCloudSurroundInd[125];

//来自odom的输入
// input: from odom
pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());

//所有可视cube点的输出
// ouput: all visualble cube points
pcl::PointCloud<PointType>::Ptr laserCloudSurround(new pcl::PointCloud<PointType>());

//围绕地图中的点来构建树
// surround points in map to build tree
pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap(new pcl::PointCloud<PointType>());

//input & output: points in one frame. local --> global
//输入和输出：一帧中的点。本地-->全局
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

//每个cube中的点
pcl::PointCloud<PointType>::Ptr laserCloudCornerArray[laserCloudNum];
pcl::PointCloud<PointType>::Ptr laserCloudSurfArray[laserCloudNum];

//kd-tree
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());

//优化的变量，是当前帧在世界坐标系下的pose
// 点云特征匹配时的优化变量
double parameters[7] = {0, 0, 0, 1, 0, 0, 0};

// Mapping线程估计的frame在world坐标系的位姿 P,因为Mapping的算法耗时很有可能会超过100ms，
// 所以这个位姿P不是实时的，LOAM最终输出的实时位姿P_realtime,需要Mapping线程计算的相对低频位姿和Odometry线程计算的相对高频位姿做整合，
// 详见后面 laserOdometryHandler 函数分析。
// 此外需要注意的是，不同于Odometry线程，这里的位姿P，即q_w_curr和t_w_curr，本身就是匹配时的优化变量。
Eigen::Map<Eigen::Quaterniond> q_w_curr(parameters);// map计算后的在world下的pose   //世界坐标系下某个点的四元数与位移
Eigen::Map<Eigen::Vector3d> t_w_curr(parameters + 4);

// 下面的两个变量是world坐标系下的Odometry计算的位姿和Mapping计算的位姿之间的增量（也即变换，transformation）
// wmap_odom * wodom_curr = wmap_curr (即前面的q/t_w_curr)
// transformation between odom's world and map's world frame
Eigen::Quaterniond q_wmap_wodom(1, 0, 0, 0);//mapping线程到Odometry线程的pose变换
Eigen::Vector3d t_wmap_wodom(0, 0, 0);//mapping线程到Odometry线程的pose变换

// Odometry线程计算的   (frame在world坐标系的)   位姿
Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr(0, 0, 0);

//接收缓存区
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::mutex mBuf;


//降采样角点和面点
pcl::VoxelGrid<PointType> downSizeFilterCorner;
pcl::VoxelGrid<PointType> downSizeFilterSurf;

//KD-tree使用的找到点的序号和距离
std::vector<int> pointSearchInd;
std::vector<float> pointSearchSqDis;

//原点和KD-tree搜索的最邻近点
PointType pointOri, pointSel;

ros::Publisher pubLaserCloudSurround, pubLaserCloudMap, pubLaserCloudFullRes, pubOdomAftMapped, pubOdomAftMappedHighFrec, pubLaserAfterMappedPath;

nav_msgs::Path laserAfterMappedPath;

//上一帧的Transform(增量)wmap_wodom   *     本帧Odometry位姿wodom_curr，     旨在为本帧Mapping位姿w_curr设置一个初始值
//里程计位姿转化为地图位姿，作为后端初始估计
void transformAssociateToMap()					//求解姿态变换矩阵
{
	q_w_curr = q_wmap_wodom * q_wodom_curr;
	t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;
}

//利用mapping计算得到的pose，计算mapping线程和Odometry线程之间的pose变换
// wmap_T_odom * odom_T_curr = wmap_T_curr
// 用在最后，当Mapping的位姿w_curr计算完毕后，更新增量wmap_wodom，旨在为下一次执行transformAssociateToMap函数时做准备
// 更新odom到map之间的位姿变换
void transformUpdate()							//进行优化后得到的新的姿态变换信息
{
	q_wmap_wodom = q_w_curr * q_wodom_curr.inverse();
	t_wmap_wodom = t_w_curr - q_wmap_wodom * t_wodom_curr;
}

//雷达坐标系点转化为地图点
// 用Mapping的位姿w_curr，将Lidar坐标系下的点变换到world坐标系下.q_w_curr为优化量，代表lidar在世界坐标系中的p
void pointAssociateToMap(PointType const *const pi, PointType *const po)
{
	Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);//雷达坐标系下的点
	Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
	po->x = point_w.x();
	po->y = point_w.y();
	po->z = point_w.z();
	po->intensity = pi->intensity;
	//po->intensity = 1.0;
}

//地图点转化到雷达坐标系点
//将map中world坐标系下的点变换到Lidar坐标系下，这个没有用到
void pointAssociateTobeMapped(PointType const *const pi, PointType *const po)
{
	Eigen::Vector3d point_w(pi->x, pi->y, pi->z);//世界坐标系下的点
	Eigen::Vector3d point_curr = q_w_curr.inverse() * (point_w - t_w_curr);
	po->x = point_curr.x();
	po->y = point_curr.y();
	po->z = point_curr.z();
	po->intensity = pi->intensity;
}

// 回调函数中将消息都是送入各自队列，进行线程加锁和解锁   三个点云的回调函数处理很简单,就是存入到各自的数据Buf中
void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudCornerLast2)
{
	mBuf.lock();//线程锁 加锁
	cornerLastBuf.push(laserCloudCornerLast2);
	mBuf.unlock();//线程锁 解锁
}

void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudSurfLast2)
{
	mBuf.lock();
	surfLastBuf.push(laserCloudSurfLast2);
	mBuf.unlock();
}

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
	mBuf.lock();
	fullResBuf.push(laserCloudFullRes2);
	mBuf.unlock();
}

//Odometry的回调函数
//接受前端发送过来的里程计算法得到帧间的位姿变换矩阵 ，    并将位姿转换到世界坐标系下后发布
// receive odomtry
/*在函数laserOdometryHandler中，将laserOdometry节点和laserMapping节点计算的位姿结合，即可得到最终的轨迹odomAftMapped*/
void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry)
{
	mBuf.lock();
	odometryBuf.push(laserOdometry);
	mBuf.unlock();

	// high frequence publish
	// 获取里程计位姿
	//将ros的消息,转成eigen的格式
	Eigen::Quaterniond q_wodom_curr;
	Eigen::Vector3d t_wodom_curr;
	q_wodom_curr.x() = laserOdometry->pose.pose.orientation.x;//接受前端发送来的 帧间的 位姿变换 ，并将其命名为q_wodom_curr、t_wodom_curr
	q_wodom_curr.y() = laserOdometry->pose.pose.orientation.y;
	q_wodom_curr.z() = laserOdometry->pose.pose.orientation.z;
	q_wodom_curr.w() = laserOdometry->pose.pose.orientation.w;
	t_wodom_curr.x() = laserOdometry->pose.pose.position.x;
	t_wodom_curr.y() = laserOdometry->pose.pose.position.y;
	t_wodom_curr.z() = laserOdometry->pose.pose.position.z;

	// Odometry的位姿，旨在用Mapping位姿的初始值（也可以理解为预测值）来实时输出，进而实现LOAM整体的实时性


	/*前端里程计会定期向后端发送位姿,这个位姿是当前帧和里程计初始位姿直接的位姿(可以叫做T_odom_curr),在后端mapping模块中,需要以map
	后端mapping模块就是解决这个问题,它会估计出里程计初始位姿和map的初始位姿的位姿变换(可以叫做T_map_odom),然后
	T_map_curr = T_map_odom×T_odom_curr*/

	Eigen::Quaterniond q_w_curr = q_wmap_wodom * q_wodom_curr;
	Eigen::Vector3d t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom; 

	// 发布出去
	nav_msgs::Odometry odomAftMapped;//q_w_curr、t_w_curr：当前转到世界坐标系下的变换矩阵
	odomAftMapped.header.frame_id = "/camera_init";
	odomAftMapped.child_frame_id = "/aft_mapped";
	odomAftMapped.header.stamp = laserOdometry->header.stamp;
	odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
	odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
	odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
	odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
	odomAftMapped.pose.pose.position.x = t_w_curr.x();
	odomAftMapped.pose.pose.position.y = t_w_curr.y();
	odomAftMapped.pose.pose.position.z = t_w_curr.z();
	pubOdomAftMappedHighFrec.publish(odomAftMapped);//转成ROS的里程计信息 发布出去
}

//进行Mapping，即帧与submap的匹配，对Odometry计算的位姿进行finetune
void process()
{
	while(1)
	{
		// 四个队列分别存放边线点、平面点、全部点、和里程计位姿，要确保需要的buffer里都有值
		// laserOdometry模块对本节点的执行频率进行了控制，laserOdometry模块publish的位姿是10Hz，点云的publish频率没这么高，限制是2hz
		while (!cornerLastBuf.empty() && !surfLastBuf.empty() &&
			!fullResBuf.empty() && !odometryBuf.empty())
		{
			mBuf.lock();//线程加锁，避免线程冲突
			// 以cornerLastBuf为基准，把时间戳小于其的全部pop出去，保证其他容器的最新消息与cornerLastBuf.front()最新消息时间戳同步
			//如果里程计信息不为空，里程计时间戳小于角特征时间戳则取出里程计数据
			while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
				odometryBuf.pop();
			//如果里程计信息为空跳出本次循环
			if (odometryBuf.empty())
			{
				mBuf.unlock();
				break;
			}

			//如果面特征信息不为空，面特征时间戳小于角特征时间戳则取出面特征数据
			while (!surfLastBuf.empty() && surfLastBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
				surfLastBuf.pop();
			if (surfLastBuf.empty())//如果面特征信息为空跳出本次循环
			{
				mBuf.unlock();
				break;
			}

			//如果全部点信息不为空，全部点云时间戳小于角特征时间戳则取出全部点云信息
			while (!fullResBuf.empty() && fullResBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
				fullResBuf.pop();
			if (fullResBuf.empty())//全部点云信息为空则跳出
			{
				mBuf.unlock();
				break;
			}
			//记录时间戳
			timeLaserCloudCornerLast = cornerLastBuf.front()->header.stamp.toSec();
			timeLaserCloudSurfLast = surfLastBuf.front()->header.stamp.toSec();
			timeLaserCloudFullRes = fullResBuf.front()->header.stamp.toSec();
			timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();
			//再次判定时间戳是否一致
			if (timeLaserCloudCornerLast != timeLaserOdometry ||
				timeLaserCloudSurfLast != timeLaserOdometry ||
				timeLaserCloudFullRes != timeLaserOdometry)
			{
				printf("time corner %f surf %f full %f odom %f \n", timeLaserCloudCornerLast, timeLaserCloudSurfLast, timeLaserCloudFullRes, timeLaserOdometry);
				printf("unsync messeage!");
				mBuf.unlock();
				break;
			}
			//清空上次角特征点云，并接收新的
			// 点云全部转成pcl的数据格式
			laserCloudCornerLast->clear();
			pcl::fromROSMsg(*cornerLastBuf.front(), *laserCloudCornerLast);
			cornerLastBuf.pop();
			//清空上次面特征点云，并接收新的
			// 点云全部转成pcl的数据格式
			laserCloudSurfLast->clear();
			pcl::fromROSMsg(*surfLastBuf.front(), *laserCloudSurfLast);
			surfLastBuf.pop();
			//清空上次全部点云，并接收新的
			// 点云全部转成pcl的数据格式
			laserCloudFullRes->clear();
			pcl::fromROSMsg(*fullResBuf.front(), *laserCloudFullRes);
			fullResBuf.pop();

			//接收里程计坐标系下的四元数与位移
			// lidar odom的结果转成eigen数据格式
			q_wodom_curr.x() = odometryBuf.front()->pose.pose.orientation.x;
			q_wodom_curr.y() = odometryBuf.front()->pose.pose.orientation.y;
			q_wodom_curr.z() = odometryBuf.front()->pose.pose.orientation.z;
			q_wodom_curr.w() = odometryBuf.front()->pose.pose.orientation.w;
			t_wodom_curr.x() = odometryBuf.front()->pose.pose.position.x;
			t_wodom_curr.y() = odometryBuf.front()->pose.pose.position.y;
			t_wodom_curr.z() = odometryBuf.front()->pose.pose.position.z;
			odometryBuf.pop();

			//角特征不为空，堆入角特征，输出目前运行实时   清空cornerLastBuf的历史缓存，为了LOAM的整体实时性
			while(!cornerLastBuf.empty())
			{
				cornerLastBuf.pop();//.pop：删除堆栈中的最新元素
				printf("drop lidar frame in mapping for real time performance \n");
			}

			mBuf.unlock();

			TicToc t_whole;//计算这个线程的全部时间

			//根据odo_to_map和point_to_odo求point_to_map
			// 上一帧的增量wmap_wodom * 本帧Odometry位姿wodom_curr，旨在为本帧Mapping位姿w_curr设置一个初始
			transformAssociateToMap();// 第一次运行时，wmap_wodom=E

			TicToc t_shift;//计算位姿转换的时间

			// 下面这是计算当前帧位置t_w_curr（在上图中用红色五角星表示的位置）IJK坐标（见上图中的坐标轴），
			// 参照LOAM_NOTED的注释，下面有关25呀，50啥的运算，等效于以50m为单位进行缩放，因为LOAM用1维数组进行cube的管理，
			//而数组的index只用是正数，所以要保证IJK坐标都是正数，所以加了laserCloudCenWidth/Heigh/Depth的偏移，
			//来使得当前位置尽量位于submap的中心处，也就是使得上图中的五角星位置尽量处于所有格子的中心附近，
			// 偏移laserCloudCenWidth/Heigh/Depth会动态调整，来保证当前位置尽量位于submap的中心处。

			//由于数组下标只能为正
			//将当前激光雷达（视角）的位置作为中心点，添加一个laserCloudCenWidth的偏移使center为正
			int centerCubeI = int((t_w_curr.x() + 25.0) / 50.0) + laserCloudCenWidth;
			int centerCubeJ = int((t_w_curr.y() + 25.0) / 50.0) + laserCloudCenHeight;
			int centerCubeK = int((t_w_curr.z() + 25.0) / 50.0) + laserCloudCenDepth;

			// 由于计算机求余是向零取整，为了不使（-50.0,50.0）求余后都向零偏移，当被求余数为负数时求余结果统一向左偏移一个单位，也即减一
			// 如果小于25就向下去整，相当于四舍五入的一个过程
			//由于int始终向0取整，所以t_w小于-25时，要修正其取整方向，使得所有取整方向一致			
			if (t_w_curr.x() + 25.0 < 0)
				centerCubeI--;
			if (t_w_curr.y() + 25.0 < 0)
				centerCubeJ--;
			if (t_w_curr.z() + 25.0 < 0)
				centerCubeK--;

			// 以下注释部分参照LOAM_NOTED，结合submap的示意图说明下面的6个while loop的作用：
			//要注意世界坐标系下的点云地图是固定的，但是IJK坐标系我们是可以移动的，
			//所以这6个while loop的作用就是调整IJK坐标系（也就是调整所有cube位置），
		    //使得五角星在IJK坐标系的坐标范围处于3 <= centerCubeI < 18， 3 < centerCubeJ < 8, 3 < centerCubeK < 18，
			//目的是为了防止后续向四周拓展cube（图中的黄色cube就是拓展的cube）时，index（即IJK坐标）成为负数。

			// 如果当前珊格索引小于3,就说明当前点快接近地图边界了，需要进行调整，相当于地图整体往x正方向移动
			while (centerCubeI < 3)
			{
				for (int j = 0; j < laserCloudHeight; j++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{ 
						int i = laserCloudWidth - 1;//指针赋值，保存最后一个指针位置
						// 从x最大值开始
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k]; 
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						//循环移位，I维度上依次后移
						for (; i >= 1; i--)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						}
						//将开始点赋值为最后一个点
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						// 该点云清零，由于是指针操作，相当于最左边的格子清空了
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}
				// 索引右移
				centerCubeI++;
				laserCloudCenWidth++;
			}
			// 同理x如果抵达右边界，就整体左移
			while (centerCubeI >= laserCloudWidth - 3)
			{ 
				for (int j = 0; j < laserCloudHeight; j++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{
						int i = 0;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; i < laserCloudWidth - 1; i++)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeI--;
				laserCloudCenWidth--;
			}
			// y和z的操作同理
			while (centerCubeJ < 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{
						int j = laserCloudHeight - 1;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; j >= 1; j--)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeJ++;
				laserCloudCenHeight++;
			}

			while (centerCubeJ >= laserCloudHeight - 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{
						int j = 0;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; j < laserCloudHeight - 1; j++)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeJ--;
				laserCloudCenHeight--;
			}

			while (centerCubeK < 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int j = 0; j < laserCloudHeight; j++)
					{
						int k = laserCloudDepth - 1;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; k >= 1; k--)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeK++;
				laserCloudCenDepth++;
			}

			while (centerCubeK >= laserCloudDepth - 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int j = 0; j < laserCloudHeight; j++)
					{
						int k = 0;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; k < laserCloudDepth - 1; k++)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeK--;
				laserCloudCenDepth--;
			}
			// 以上操作相当于维护了一个局部地图，保证当前帧不在这个局部地图的边缘，这样才可以从地图中获取足够的约束

			int laserCloudValidNum = 0;
			int laserCloudSurroundNum = 0;
			// 从当前格子为中心，选出地图中一定范围的点云
			// 即向IJ坐标轴的正负方向各拓展2个栅格，K坐标轴的正负方向各拓展1个栅格
			// 在每一维附近5个栅格(前2个，后2个，中间1个)里进行查找（前后250米范围内，总共500米范围），三个维度总共125个栅格
			// 在这125个栅格里面进一步筛选在视域范围内的栅格

			for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++)
			{
				for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++)
				{
					for (int k = centerCubeK - 1; k <= centerCubeK + 1; k++)
					{
						// 如果坐标合理
						if (i >= 0 && i < laserCloudWidth &&
							j >= 0 && j < laserCloudHeight &&
							k >= 0 && k < laserCloudDepth)
						{ 
							//记住视域范围内的cube索引，匹配用	laserCloudWidth=21 laserCloudHeight=21
							laserCloudValidInd[laserCloudValidNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
							laserCloudValidNum++;
							//记住附近所有cube的索引，显示用	
							laserCloudSurroundInd[laserCloudSurroundNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
							laserCloudSurroundNum++;
						}
					}
				}
			}

			laserCloudCornerFromMap->clear();
			laserCloudSurfFromMap->clear();
			//将有效栅格的点云叠加到一起组成submap子地图的特征点云，构建用来这一帧优化的局部地图
			for (int i = 0; i < laserCloudValidNum; i++)
			{
				*laserCloudCornerFromMap += *laserCloudCornerArray[laserCloudValidInd[i]];
				*laserCloudSurfFromMap += *laserCloudSurfArray[laserCloudValidInd[i]];
			}
			int laserCloudCornerFromMapNum = laserCloudCornerFromMap->points.size();
			int laserCloudSurfFromMapNum = laserCloudSurfFromMap->points.size();

// -------------------至此，得到当前帧的局部地图的特征点云-------------------

			// 为了减少运算量，对点云进行降采样（此次的下采样是对当前帧的下采样，并非地图的下采样）
			pcl::PointCloud<PointType>::Ptr laserCloudCornerStack(new pcl::PointCloud<PointType>());//对
			downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
			downSizeFilterCorner.filter(*laserCloudCornerStack);
			int laserCloudCornerStackNum = laserCloudCornerStack->points.size();

			pcl::PointCloud<PointType>::Ptr laserCloudSurfStack(new pcl::PointCloud<PointType>());
			downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
			downSizeFilterSurf.filter(*laserCloudSurfStack);
			int laserCloudSurfStackNum = laserCloudSurfStack->points.size();

			printf("map prepare time %f ms\n", t_shift.toc());//打印位姿转换的时间
			printf("map corner num %d  surf num %d \n", laserCloudCornerFromMapNum, laserCloudSurfFromMapNum);//打印地图中角点和平面点的数量

			//下面是后端Scan to Map的匹配优化。Submap子地图的网格与全部地图的网格进行匹配时
			//这里的匹配方法如下
			//1. 取当前帧的特征点（边线点/平面点）
			//2. 找到全部地图特征点中，当前特征点的5个最近邻点。
			//3. 如果是边线点，则以这五个点的均值点为中心，以5个点的主方向向量（类似于PCA方法）为方向，
			//作一条直线，令该边线点与直线距离最短，构建非线性优化问题。
			//4. 如果是平面点，则寻找五个点的法方向（反向的PCA方法），
			//令这个平面点在法方向上与五个近邻点的距离最小，构建非线性优化问题。
			//5. 优化变量是雷达位姿，求解能够让以上非线性问题代价函数最小的雷达位姿

			// 最终的地图有效点云数目进行判断
			if (laserCloudCornerFromMapNum > 10 && laserCloudSurfFromMapNum > 50)
			{
				TicToc t_opt;//计算优化时间
				TicToc t_tree;//计算KD-tree搜索时间
				// 送入kdtree便于最近邻搜索
				kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMap);
				kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMap);
				printf("build tree time %f ms \n", t_tree.toc());
				//优化两次，第二次在第一次得到的pose上进行
				for (int iterCount = 0; iterCount < 2; iterCount++)
				{
					// 建立ceres问题
					//ceres::LossFunction *loss_function = NULL;
					ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
					ceres::LocalParameterization *q_parameterization =
						new ceres::EigenQuaternionParameterization();
					ceres::Problem::Options problem_options;

					ceres::Problem problem(problem_options);
					problem.AddParameterBlock(parameters, 4, q_parameterization);
					problem.AddParameterBlock(parameters + 4, 3);

					TicToc t_data;//计算建图数据点关联的时间
					int corner_num = 0;
					// 构建边线点（角点）相关的约束

					for (int i = 0; i < laserCloudCornerStackNum; i++)
					{
						pointOri = laserCloudCornerStack->points[i];
						//double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;


						// 需要注意的是submap中的点云都是world坐标系，而当前帧的点云都是Lidar坐标系，所以
						// 在搜寻最近邻点时，先用预测的Mapping位姿w_curr，将Lidar坐标系下的特征点变换到world坐标系下
						// 把当前点根据初值投到地图坐标系下去
						
						pointAssociateToMap(&pointOri, &pointSel);
						// 地图中寻找和该点最近的5个点
						// 在submap的corner特征点（target）中，寻找距离当前帧corner特征点（source）最近的5个点
						kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis); 
						// 判断最远的点距离不能超过1m，否则就是无效约束

						if (pointSearchSqDis[4] < 1.0)
						{ 
							// 计算这个5个最近邻点的中心
							std::vector<Eigen::Vector3d> nearCorners;
							Eigen::Vector3d center(0, 0, 0);
							for (int j = 0; j < 5; j++)
							{
								Eigen::Vector3d tmp(laserCloudCornerFromMap->points[pointSearchInd[j]].x,
													laserCloudCornerFromMap->points[pointSearchInd[j]].y,
													laserCloudCornerFromMap->points[pointSearchInd[j]].z);
								center = center + tmp;
								nearCorners.push_back(tmp);
							}
							//计算这五个点的均值
							center = center / 5.0;

							// 计算这个5个最近邻点的协方差矩阵
							Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
							for (int j = 0; j < 5; j++)
							{
								Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
								covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
							}

							// 进行特征值分解
							//计算协方差矩阵的特征值和特征向量，用于判断这5个点是不是呈线状分布，此为PCA的原理
							Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

							// if is indeed line feature
							// note Eigen library sort eigenvalues in increasing order
							// PCA的原理：计算协方差矩阵的特征值和特征向量，用于判断这5个点是不是呈线状分布
							// 根据特征值分解情况看看是不是真正的线特征
							// 特征向量就是线特征的方向
							Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
							// 如果5个点呈线状分布，最大的特征值对应的特征向量就是该线的方向向量
							Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
							//如果最大的特征值 >> 其他特征值，则5个点确实呈线状分布，否则认为直线“不够直”
							if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])// 最大特征值大于次大特征值的3倍认为是线特征
							{ 
								Eigen::Vector3d point_on_line = center;
								Eigen::Vector3d point_a, point_b;
								// 根据拟合出来的线特征方向，以平均点为中心构建两个虚拟点，代替一条直线
								// 从中心点沿着方向向量向两端移动0.1m，使用两个点代替一条直线，
								//这样计算点到直线的距离的形式就跟laserOdometry中保持一致
								point_a = 0.1 * unit_direction + point_on_line;
								point_b = -0.1 * unit_direction + point_on_line;
								// 这里点到线的ICP过程就比Odometry中的要鲁棒和准确一些了（当然也就更耗时一些）
								// 因为不是简单粗暴地搜最近的两个corner点作为target的线，
								//而是PCA计算出最近邻的5个点的主方向作为线的方向，而且还会check直线是不是“足够直”

								// 构建约束，和lidar odom约束一致
								ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, point_a, point_b, 1.0);
								problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
								corner_num++;	
							}							
						}
						/*
						else if(pointSearchSqDis[4] < 0.01 * sqrtDis)
						{
							Eigen::Vector3d center(0, 0, 0);
							for (int j = 0; j < 5; j++)
							{
								Eigen::Vector3d tmp(laserCloudCornerFromMap->points[pointSearchInd[j]].x,
													laserCloudCornerFromMap->points[pointSearchInd[j]].y,
													laserCloudCornerFromMap->points[pointSearchInd[j]].z);
								center = center + tmp;
							}
							center = center / 5.0;	
							Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
							ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, center);
							problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
						}
						*/
					}
					//根据法线判断是否为面特征
					int surf_num = 0;
					for (int i = 0; i < laserCloudSurfStackNum; i++)
					{
						pointOri = laserCloudSurfStack->points[i];
						//double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;			
						pointAssociateToMap(&pointOri, &pointSel);
						kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

						// 求面的法向量就不是用的PCA了（虽然论文中说还是PCA），使用的是最小二乘拟合
						// 假设平面不通过原点，则平面的一般方程为Ax + By + Cz + 1 = 0，用这个假设可以少算一个参数，提效
						Eigen::Matrix<double, 5, 3> matA0;
						Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
						// 构建平面方程Ax + By +Cz + 1 = 0
						// 通过构建一个超定方程来求解这个平面方程
						if (pointSearchSqDis[4] < 1.0)
						{
							for (int j = 0; j < 5; j++)
							{
								matA0(j, 0) = laserCloudSurfFromMap->points[pointSearchInd[j]].x;
								matA0(j, 1) = laserCloudSurfFromMap->points[pointSearchInd[j]].y;
								matA0(j, 2) = laserCloudSurfFromMap->points[pointSearchInd[j]].z;
							}
							// 调用eigen接口求解该方程，解就是这个平面的法向量
							Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
							
							double negative_OA_dot_norm = 1 / norm.norm();// 法向量长度的倒数
							norm.normalize();// 法向量归一化

							// Here n(pa, pb, pc) is unit norm of plane
							bool planeValid = true;
							// 根据求出来的平面方程进行校验，看看是不是符合平面约束
							for (int j = 0; j < 5; j++)
							{
								// if OX * n > 0.2, then plane is not fit well
								// 这里是求解点到平面的距离
								// 点(x0, y0, z0)到平面Ax + By + Cz + D = 0 的距离公式 = fabs(Ax0 + By0 + Cz0 + D) / sqrt(A^2 + B^2 + C^2)
								if (fabs(norm(0) * laserCloudSurfFromMap->points[pointSearchInd[j]].x +
										 norm(1) * laserCloudSurfFromMap->points[pointSearchInd[j]].y +
										 norm(2) * laserCloudSurfFromMap->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
								{
									planeValid = false;// 点如果距离平面太远，就认为这是一个拟合的不好的平面
									break;
								}
							}
							Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
							// 如果平面有效就构建平面约束
							if (planeValid)
							{
								// 利用平面方程构建约束，和前端构建形式稍有不同
								ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(curr_point, norm, negative_OA_dot_norm);
								problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
								surf_num++;
							}
						}
						/*
						else if(pointSearchSqDis[4] < 0.01 * sqrtDis)
						{
							Eigen::Vector3d center(0, 0, 0);
							for (int j = 0; j < 5; j++)
							{
								Eigen::Vector3d tmp(laserCloudSurfFromMap->points[pointSearchInd[j]].x,
													laserCloudSurfFromMap->points[pointSearchInd[j]].y,
													laserCloudSurfFromMap->points[pointSearchInd[j]].z);
								center = center + tmp;
							}
							center = center / 5.0;	
							Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
							ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, center);
							problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
						}
						*/
					}

					//printf("corner num %d used corner num %d \n", laserCloudCornerStackNum, corner_num);
					//printf("surf num %d used surf num %d \n", laserCloudSurfStackNum, surf_num);

					printf("mapping data assosiation time %f ms \n", t_data.toc());
					// 调用ceres求解
					TicToc t_solver;
					ceres::Solver::Options options;
					options.linear_solver_type = ceres::DENSE_QR;
					options.max_num_iterations = 4;
					options.minimizer_progress_to_stdout = false;
					options.check_gradients = false;
					options.gradient_check_relative_precision = 1e-4;
					ceres::Solver::Summary summary;
					ceres::Solve(options, &problem, &summary);
					printf("mapping solver time %f ms \n", t_solver.toc());	//打印解算时间  t_solver

					//printf("time %f \n", timeLaserOdometry);
					//printf("corner factor num %d surf factor num %d\n", corner_num, surf_num);
					//printf("result q %f %f %f %f result t %f %f %f\n", parameters[3], parameters[0], parameters[1], parameters[2],
					//	   parameters[4], parameters[5], parameters[6]);
				}
				printf("mapping optimization time %f \n", t_opt.toc());//打印建图数据关联时间
			}
			else
			{
				ROS_WARN("time Map corner and surf num are not enough");
			}

			//更新mapping到Odometry线程的T
			// 完成ICP（迭代2次）的特征匹配后，用最后匹配计算出的优化变量 w_curr，更新增量wmap_wodom，为下一次Mapping做准备
			
			//迭代结束更新相关的转移矩阵
			transformUpdate();


			TicToc t_add;//计算增加特征点的时间

			//下面是一些后处理的工作，即将当前帧的特征点加入到全部地图栅格中，对全部地图栅格中的点进行降采样，
			//刷新附近点云地图，刷新全部点云地图，发布当前帧的精确位姿和平移估计
			
			// 下面两个for loop的作用就是将当前帧的特征点云，逐点进行操作：转换到world坐标系并添加到对应位置的cube中
			// 将优化后的当前帧边线点（角点）加到对应的边线点局部地图中去

			// 将当前帧的（次极大边线点云，经过降采样后的）存入对应的边线点云的cube
			for (int i = 0; i < laserCloudCornerStackNum; i++)
			{
				pointAssociateToMap(&laserCloudCornerStack->points[i], &pointSel);//把当前帧（已经过下采样）的点云转移到世界坐标系下---->pointSel

				//按50的比例尺缩小，四舍五入，偏移laserCloudCen*的量，计算索引	
				// 计算本次的特征点的IJK坐标，进而确定添加到哪个cube中
				int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;//laserCloudCenWidth=10
				int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;//laserCloudCenHeight=10
				int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;//laserCloudCenDepth=5
				
				// 同样四舍五入一下
				if (pointSel.x + 25.0 < 0)
					cubeI--;
				if (pointSel.y + 25.0 < 0)
					cubeJ--;
				if (pointSel.z + 25.0 < 0)
					cubeK--;

				//只挑选-laserCloudCenWidth * 50.0 < point.x < laserCloudCenWidth * 50.0范围内的点，y和z同理
				// 如果超过边界的话就算了
				//按照尺度放进不同的组，每个组的点数量各异
				if (cubeI >= 0 && cubeI < laserCloudWidth &&
					cubeJ >= 0 && cubeJ < laserCloudHeight &&
					cubeK >= 0 && cubeK < laserCloudDepth)
				{
					// 根据xyz的索引计算在一位数组中的索引
					int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
					laserCloudCornerArray[cubeInd]->push_back(pointSel);//根据索引将当前帧特征点加入到局部地图中去
				}
			}

			// 面点也做同样的处理
			// 将当前帧的（次极小平面点云，经过降采样后的）存入对应的平面点云的cube
			for (int i = 0; i < laserCloudSurfStackNum; i++)
			{
				pointAssociateToMap(&laserCloudSurfStack->points[i], &pointSel);

				int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
				int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
				int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

				if (pointSel.x + 25.0 < 0)
					cubeI--;
				if (pointSel.y + 25.0 < 0)
					cubeJ--;
				if (pointSel.z + 25.0 < 0)
					cubeK--;

				if (cubeI >= 0 && cubeI < laserCloudWidth &&
					cubeJ >= 0 && cubeJ < laserCloudHeight &&
					cubeK >= 0 && cubeK < laserCloudDepth)
				{
					int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
					laserCloudSurfArray[cubeInd]->push_back(pointSel);
				}
			}
			printf("add points time %f ms\n", t_add.toc());//打印增加特征点的时间

			
			TicToc t_filter;//计算降采样的时间

			// 因为新增加了点云，对之前已经存有点云的submap cube全部重新进行一次降采样
			// 这个地方可以简单优化一下：如果之前的cube没有新添加点就不需要再降采样
			// 这可以通过记录上一个循环中存入对应cubeInd的idx来实现
			for (int i = 0; i < laserCloudValidNum; i++)
			{
				int ind = laserCloudValidInd[i];// submap中每一个cube的索引
				// 判断当前的submap的cube id 是否在当前帧的索引的vector中
				pcl::PointCloud<PointType>::Ptr tmpCorner(new pcl::PointCloud<PointType>());
				downSizeFilterCorner.setInputCloud(laserCloudCornerArray[ind]);
				downSizeFilterCorner.filter(*tmpCorner);
				laserCloudCornerArray[ind] = tmpCorner;

				pcl::PointCloud<PointType>::Ptr tmpSurf(new pcl::PointCloud<PointType>());
				downSizeFilterSurf.setInputCloud(laserCloudSurfArray[ind]);
				downSizeFilterSurf.filter(*tmpSurf);
				laserCloudSurfArray[ind] = tmpSurf;
			}
			printf("filter time %f ms \n", t_filter.toc());//打印降采样的时间
			
			TicToc t_pub;//计算发布地图话题数据的时间

			//publish surround map for every 5 frame
			// 每隔5帧对外发布一下
			if (frameCount % 5 == 0)
			{
				laserCloudSurround->clear();
				// 把该当前帧相关的局部地图发布出去
				for (int i = 0; i < laserCloudSurroundNum; i++)
				{
					int ind = laserCloudSurroundInd[i];
					*laserCloudSurround += *laserCloudCornerArray[ind];
					*laserCloudSurround += *laserCloudSurfArray[ind];
				}

				sensor_msgs::PointCloud2 laserCloudSurround3;
				pcl::toROSMsg(*laserCloudSurround, laserCloudSurround3);
				laserCloudSurround3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
				laserCloudSurround3.header.frame_id = "/camera_init";
				pubLaserCloudSurround.publish(laserCloudSurround3);
			}

			// 每隔20帧发布全量的局部地图
			// 每20帧发布IJK局部地图
			if (frameCount % 20 == 0)
			{
				pcl::PointCloud<PointType> laserCloudMap;
				for (int i = 0; i < 4851; i++)
				{
					laserCloudMap += *laserCloudCornerArray[i];
					laserCloudMap += *laserCloudSurfArray[i];
				}
				sensor_msgs::PointCloud2 laserCloudMsg;
				pcl::toROSMsg(laserCloudMap, laserCloudMsg);
				laserCloudMsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
				laserCloudMsg.header.frame_id = "/camera_init";
				pubLaserCloudMap.publish(laserCloudMsg);
			}

			// 全部点云转化到world坐标系，并发布
			int laserCloudFullResNum = laserCloudFullRes->points.size();
			// 把当前帧发布出去
			for (int i = 0; i < laserCloudFullResNum; i++)
			{
				pointAssociateToMap(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
			}

			sensor_msgs::PointCloud2 laserCloudFullRes3;
			pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
			laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
			laserCloudFullRes3.header.frame_id = "/camera_init";
			pubLaserCloudFullRes.publish(laserCloudFullRes3);

			printf("mapping pub time %f ms \n", t_pub.toc());//打印发布地图话题数据的时间

			printf("whole mapping time %f ms +++++\n", t_whole.toc());//打印在全部运行的时间

			// 发布当前位姿
			nav_msgs::Odometry odomAftMapped;
			odomAftMapped.header.frame_id = "/camera_init";
			odomAftMapped.child_frame_id = "/aft_mapped";
			odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
			odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
			odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
			odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
			odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
			odomAftMapped.pose.pose.position.x = t_w_curr.x();
			odomAftMapped.pose.pose.position.y = t_w_curr.y();
			odomAftMapped.pose.pose.position.z = t_w_curr.z();
			pubOdomAftMapped.publish(odomAftMapped);

			// 发布当前轨迹
			geometry_msgs::PoseStamped laserAfterMappedPose;
			laserAfterMappedPose.header = odomAftMapped.header;
			laserAfterMappedPose.pose = odomAftMapped.pose.pose;
			laserAfterMappedPath.header.stamp = odomAftMapped.header.stamp;
			laserAfterMappedPath.header.frame_id = "/camera_init";
			laserAfterMappedPath.poses.push_back(laserAfterMappedPose);
			pubLaserAfterMappedPath.publish(laserAfterMappedPath);

			
			// 发布tf
			static tf::TransformBroadcaster br;
			tf::Transform transform;
			tf::Quaternion q;
			transform.setOrigin(tf::Vector3(t_w_curr(0),
											t_w_curr(1),
											t_w_curr(2)));
			q.setW(q_w_curr.w());
			q.setX(q_w_curr.x());
			q.setY(q_w_curr.y());
			q.setZ(q_w_curr.z());
			transform.setRotation(q);
			br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "/camera_init", "/aft_mapped"));
			//br.sendTransform(tf::StampedTransform(A,B,C,D))作用是调用sendTransform()函数，向系统中广播参考系之间的坐标变换关系，
			//其中所广播的变换关系的数据类型是tf::StampedTransform，它包含了四个参数ABCD
			//第一个参数是存储坐标系之间变换关系的变量	transform.setOrigin             transform.setRotation
			//第二个参数是广播TF变换关系的时间戳
			//第三个参数是传递的父节点坐标系的名称，即/camera_init
			//第四个参数是传递的子节点坐标系的名称，即/aft_mapped
			frameCount++;
		}
		std::chrono::milliseconds dura(2);//延时2ms
        std::this_thread::sleep_for(dura);
	}
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "laserMapping");
	ros::NodeHandle nh;
	//配置线特征和面特征点云的分辨率,之后通过体素滤波进行下采样
	float lineRes = 0;//声明线特征分辨率
	float planeRes = 0;//声明面特征分辨率
	nh.param<float>("mapping_line_resolution", lineRes, 0.4);//从参数服务器读取线特征分辨率			//通过lineRes给mapping_line_resolution参数赋值
	nh.param<float>("mapping_plane_resolution", planeRes, 0.8);//从参数服务器读取面特征分辨率
	printf("line resolution %f plane resolution %f \n", lineRes, planeRes);//打印设置的分辨率
	downSizeFilterCorner.setLeafSize(lineRes, lineRes,lineRes);//设置线特征的分辨率					//进行体素滤波实现降采样
	downSizeFilterSurf.setLeafSize(planeRes, planeRes, planeRes);//设置面特征的分辨率

	//订阅4种消息
	//上一帧的角点	// 从laserOdometry节点接收次极大边线点
	ros::Subscriber subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100, laserCloudCornerLastHandler);
	//上一帧的面点	// 从laserOdometry节点接收次极小平面点
	ros::Subscriber subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100, laserCloudSurfLastHandler);
	//前端里程计计算得到的位姿	// 从laserOdometry节点接收到的最新帧的位姿T_cur^w
	ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 100, laserOdometryHandler);
	//所有的点云	// 从laserOdometry节点接收到的当前帧原始点云（只经过一次降采样）
	ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100, laserCloudFullResHandler);
	
	//继续回到main函数中之后是定义本节点发布的消息(6种)
	
	//注册发布点云
	// submap（子地图）所在cube（栅格）中的点云，发布周围5帧的点云（降采样以后的）	
	pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 100);
	//map地图
	pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 100);
	// 当前帧原始点云
	pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 100);
	//经过Map to Map精估计优化后的当前帧位姿																	发布当前帧的精优变换！
	pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 100);
	// 将里程计坐标系位姿转化到世界坐标系位姿（地图坐标系），相当于位姿优化初值，即Odometry odom 到  map
	pubOdomAftMappedHighFrec = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init_high_frec", 100);
	// 经过Map to Map精估计优化后的当前帧平移																	发布当前帧的精优变换
	pubLaserAfterMappedPath = nh.advertise<nav_msgs::Path>("/aft_mapped_path", 100);

	//地图的数组进行 重置 分配地址
	for (int i = 0; i < laserCloudNum; i++)
	{
		//后端地图的数组 用栅格的形式

		//laserCloudCornerArray[5]中存放的数据为局部地图中第5个珊格中存放的是某个角点的世界坐标系信息		
		laserCloudCornerArray[i].reset(new pcl::PointCloud<PointType>());
		laserCloudSurfArray[i].reset(new pcl::PointCloud<PointType>());
	}

	//主执行程序
	std::thread mapping_process{process};//process是一个 while(1)的循环

	ros::spin();//不断执行回调函数

	return 0;
}
