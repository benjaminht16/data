arXiv:2306.02057v1 [eess.SP] 3 Jun 2023
DataAI-6G: A System Parameters Configurable
Channel Dataset for AI-6G Research
Zibing Shen, Jianhua Zhang, Li Yu, Yuxiang Zhang, Zhen Zhang, Xidong Hu
State Key Laboratory of Networking and Switching Technology, Beijing University of Posts and Telecommunications,
Beijing, China
Email: {szb, jhzhang, li.yu, zhangyx, zhenzhang, hxd}@bupt.edu.cn
Abstract—With the acceleration of the commercialization of
fifth generation (5G) mobile communication technology and the
research for 6G communication systems, the communication
system has the characteristics of high frequency, multi-band, high
speed movement of users and large antenna array. These bring
many difficulties to obtain accurate channel state information
(CSI), which makes the performance of traditional communication
methods be greatly restricted. Therefore, there has been
a lot of interest in using artificial intelligence (AI) instead of
traditional methods to improve performance. A common and
accurate dataset is essential for the research of AI communication.
However, the common datasets nowadays still lack
some important features, such as mobile features, spatial nonstationary
features etc. To address these issues, we give a dataset
for future 6G communication. In this dataset, we address these
issues with specific simulation methods and accompanying code
processing.
Index Terms—AI, mobile features, spatial non-stationary features
I. INTRODUCTION
6G mobile networks are expected to support further enhanced
mobile broadband, ultramassive machine-type, enhanced
ultrareliable and low-latency, long-distance, and highmobility
communications and other emerging scenarios for the
2030 intelligent information society, which requires instantaneous,
extremely high-speed wireless connectivity [1]. These
new scenarios and requirements make it necessary to consider
an increasing number of features when modeling the channel.
The channel model based on statistical characteristics becomes
more and more complex as the number of characteristics
considered increases, and an overly complex model is not
conducive to future research. In order not to further complicate
the model, researchers have come up with the idea of using
AI techniques instead of, or in addition to, optimizing the
traditional modeling approach. This idea is well supported in
today’s era of big data.
With the continuous exploration of researchers, machine
learning (ML)- based AI techniques have become the key to
develop the next-generation communication system [2]. Highmobility
communications make the CSI tends to be out of
date in a short time period, multi-antenna and multi-band
make acquiring CSI difficult and requires significant overhead,
and with the usage of ultramassive MIMO, the energy
consumed by signal transmission and RF chains will become
considerable. These make it very difficult to obtain channel
information in the space, time, and frequency domains. In
order to get accurate CSI and reduce overhead, AI-based time-
, frequency-, and space- domain channel extrapolation [3] and
compressive sensing for massiveMIMO CSI feedback [4] have
been presented. In the millimeter wave band, blocking has a
significant impact on the quality of communication and the
overhead of beam selection is huge, which are the challenges
of future high frequency communication. In [5], [6], two AIbased
methods for blockage prediction and beam prediction
are proposed, and both of these methods effectively solve the
above problems. In addition, the prediction of a particular
channel characteristic, such as path loss [7], can also be very
useful to further improve the communication quality.
Fig. 1. Communication with AI.
Adding AI at the base station and user side can improve
the performance and reduce the overhead of communication.
And to implement these AI applications, a large amount
of channel data is necessary. The set of these data is the
essential channel dataset in AI training, as shown in Fig. 1.
To meet the data requirements of the researcher, we have
introduced the DataAI-6G dataset, which is designed for
machine learning research in wireless channel transmission
and modeling.More specifically, using this dataset, researchers
can easily construct the inputs and outputs of several machine
learning applications. The DataAI-6G dataset provides angle
of departure (AOD), angle of arrival (AOA), delay, phase,
power of each path and the path loss between any pair of
transceiver antennas. These data are obtained from the raytracing
simulator, Wireless InSite, developed by Remcom [8].
Remcom Wireless InSite, is widely used in mmWave and
massive MIMO research at both industry and academia, and
has been verified with real-world channel measurements [9]–
[11]. More important, our dataset has a dedicated set of codes,
which can synthesize the UL and DL channel matrices and has
user moving function. More details will be discussed in the
rest of this paper.
II. DESIGN OF DATASET
According to [12], the use of ultra-large antenna arrays
introduces near-field spatial non-stationary features, which
is an essential feature in 6G communications. In [13], the
existence of UL to DL mapping has been proved, so there
are many researchers are studying the mapping of UL to DL.
In the future wireless communication, high-speed movement
features are the focus of attention, but users in existing datasets
are usually static. So, in our dataset, we have considered the
above three features. Firstly, in the data simulation stage,
we simulate each antenna array element separately instead
of using the plane wave synthesis method. Then in the the
code synthesis stage, the dataset can synthesize the CSI of
the UL/DL channel using the angle, delay, power, and phase
information of each path. More importantly, the dataset is able
to introduce further Doppler phase shifts on this basis to obtain
the CSI of the moving state. The specific synthesis principle
is as follows.
In our dataset, multi-antenna technology has been considered.
In order to get the channel matrix, we need calculate the
channel impulse response (CIR) of each antenna pair first.
Consider a MIMO system with multiple base stations and
multiple user areas. For the k-th antenna at the base station
x and the g-th antenna at the u-th user point in the y-th user
area, there will be a large number of Multipath components
(MPC) between them. So the CIR can be written as
hxk,yug =
M
X i=1
iej'i( − i), (1)
where i and 'i represent the amplitude and the phase of the
i-th path, respectively. M denotes the total number of paths
between these two antennas and i denotes the delay of the
i-th path.
However, in real-world communication, the receiving antenna
usually samples the received signal at a certain frequency,
so the received signal will be divided into multiple
time-delayed distinguishable paths. In the DataAI-6G dataset,
we simulate this reception method to obtain the channel
response that most closely resembles the actual situation.
Assuming that the receiver samples the received signal at a
sampling interval of 1/BW (BW is the channel bandwidth),
the channel impulse response at the i-th sampling interval can
be expressed as
hi
xk,yug = (
Ni
X
n=1
nej'n)( − i), (2)
where n and 'n represent the amplitude and the phase of the
n-th path, respectively. Ni denotes the total number of paths
in the i-th sampling interval and i denotes the delay of the
i-th sampling interval.
After that, the impulse responses of all sampling intervals
are superimposed and converted to the frequency domain
to obtain the frequency domain channel response. Then, the
user-set UL/DL carrier frequency is brought into the formula
of frequency domain channel response to obtain a complex
value, and this complex value is stored as an approximate
channel response in the generated dataset. The UL/DL channel
response can be written as

Hup
xk,yug = PL
i=1(PNi
n=1 nej'n)e−j2fupl ,
Hdown
xk,yug = PL
i=1(PNi
n=1 nej'n)e−j2fdownl ,
(3)
where L denotes the number of sampling intervals and
fup/fdown denotes the UL/DL carrier frequency. Due to the
difference of UL and DL carrier frequencies, the UL and DL
channel response will be different in magnitude and phase.
And since the UL and DL channel responses are calculated
using similar formulas, there is a strong correlation between
them. The advantage of using this approach to obtain the
UL and DL channel responses is that the researcher has the
flexibility to set the UL and DL carrier frequencies. However,
since only the simulation data of the DL channel are available,
this synthesis can only approximate the UL channel, which
is still lacking in terms of accuracy. To further improve the
accuracy of the UL and DL channels, UL simulation data or
actual measurement data can be included in future studies.
On the basis of the UL/DL features, we will proceed to
discuss the mobile features. To get the mobile features, we
need to take Doppler phase shift into consider. The expression
of the Doppler phase shift can be written as
' = 2
v · n
c
t, (4)
where v denotes the velocity vector in the direction of movement
and n denotes the direction vector of AOA in DL channel
and negative direction vector of AOD in UL channel. c and
t represents the wavelength of the carrier wave and time
interval, respectively.
After get the Doppler phase shift, we add it to Eq.(3). For
the k-th antenna at the base station x and the g-th antenna at
the u-th user point in the y-th user area, the frequency domain
channel response in the moving state can be written as

Hup
xk,yug = PL
i=1(PNi
n=1 nej('n+'))e−j2fupl .
Hdown
xk,yug = PL
i=1(PNi
n=1 nej('n+'))e−j2fdownl .
(5)
The channel response obtained in this way contains mobile
features, so our dataset is well suited to researchers for the
study of mobile features.
III. DATASET GENERATION
We build a large outdoor street scenario with multiple
configurations in multiple bands using Wireless Insite [8] and
simulate it to obtain a set of channel parameters. We provide
a generic framework that allows researchers the flexibility
Fig. 2. Framework of the dataset.
Fig. 3. Outdoor street scenario.
to configure some parameters in the code according to their
needs. As shown in Fig. 2, the researchers can then bring the
raw channel parameters as input to the framework to output
the customized dataset.
A. Outdoor street scenario
The outdoor street scenario is dedicated to providing researchers
with diverse scene features to meet the needs of
machine learning different requirements. The whole scenario
is 646 m long and 290 m wide, which is an extensive outdoor
scene, as shown in Fig. 3. Two horizontally oriented main
streets run through the whole scenario, and four vertically
oriented secondary streets are connected to the horizontally
oriented ones. To provide multi-regionalized data, we set up
at least one base station for each street. In total, we build 8
BS and 12 user grids, which are scattered within 6 streets.
The users on the streets are evenly distributed within the grid.
In addition, the streets are flanked by buildings of different
heights and vegetation of varying sizes. For simplicity, the
buildings are rectangular and solid, so that the rays from the
base station cannot penetrate the buildings.
More detail, the locations of these 8 BS are distributed on
both sides of the street. Four of the base stations are set up
in two main streets in a horizontal direction, and four base
stations are respectively set up in four streets in a vertical
direction. Each base station is equipped with different types
of antennas as well as different heights. TX2 and TX5 are
equipped with a single element, which is the omnidirectional
antenna, and the rest of the base stations are equipped with
multiple antennas. It is necessary to elaborate that each array
element constituting the MIMO antenna array is a half-wave
dipole, and the distance between them is half a wavelength.
Users are evenly distributed among 12 user grids, and each
starting point of the user grid is located in the left corner. An
example of the user points arrangement is shown in Fig. 4.
The users in RX1, RX2 and RX3 are equipped with a 2×2
uniform planar array, and the other users in the rest of the user
gird are equipped with a omnidirectional antenna.
Fig. 4. User points arrangement.
We use the X3D model, which is by far the most versatile,
functional, and accurate propagation model in Wireless Insite.
Considering the meaningful received power, for simplicity,
only the first 4 reflections are considered. More importantly,
the accuracy of blocking and beam prediction can be further
improved by exploiting the diffraction properties [5], [6]. But
on the other hand, the received power decreases significantly
as the number of diffractions increases, so we turn on only
one diffraction. After we configure the main parameters in
Wireless Insite, it performs signal propagation simulation and
finally gives ray tracing results. The results of the simulation
contain (i) the azimuth and elevation angles of departure of
each path, (ii) the azimuth and elevation angles of arrival of
each path, (iii) the path receive power, (iv) the path phase and
(v) the propagation delay of each path. Wireless Insite can
also output the overall received power, the overall phase, and
the path loss of a receive point for all valid paths.
B. Advantages of dataset
Compared with other datasets, such as DeepMIMO [14],
Wireless AI Research Dataset [15]. Our dataset has three major
advantages(as shown in Table 1): (i) Spatial non-stationary
features are considered in the simulation. (ii) With the user
moving function that considers Doppler, users can freely
configure the moving route and moving speed. (iii) Has the
ability to generate any number of user points. In the next of
this part, we will explain in detail how the last two functions
are implemented in the code.
In the profile of the code, the researcher can select the
base station and the user area to be activated, and can
also select the desired user points in the user area, while
the number of antennas of users and base stations can be
freely set according to the requirements. After setting the
above parameters, researchers can choose the frequency of the
channel(3.5 GHz, 28 GHz or 60 GHz), the antenna pattern of
users and base stations, the bandwidth and carrier frequencies
of UL/DL channel. Then the code will extract the AOA, AOD,
power, delay and phase of each path, and the path loss of the
channel will also be extracted. After obtained the angle, phase,
delay and power information of each path, The dataset will
synthesize the channel response using Eq.(3).
If researchers want the user to move in the user grid, they
just need set the parameter ’move’ to ’t’. In the DataAI-6G
dataset, the user can move along four directions: up, down, left
and right . The researcher only needs to set the corresponding
parameters in the configuration file to specify both the path
and direction of movement. Then, the code will perform point
sampling on the movement path according to the user-set speed
and sampling interval. In order to calculate the Doppler phase
shift due to the movement, the distance difference between the
virtual point that get by point sampling and the user point in
the user gird has to be calculated first.
d = vt − ms, (6)
where  means the -th sample point, s denotes the interval
between real user points, v and t represent the speed of the
user and the sampling interval. m in Eq.(6) means the m-th
user point in the moving path, which calculated by ⌈ vt
s −
1⌉. Then, the code will bring d into Eq.(4) to calculate the
Doppler phase shift, which can be written as
' = 2d
m· n
c
, (7)
where m denotes the unit vector in the direction of movement
and n denotes the direction vector of AOA/AOD in DL/UL
channel. c represents the wavelength of the carrier wave.
After get the Doppler phase shift, the dataset will synthesize
the channel response in the moving state using Eq.(5).
IV. CASE OF BEAM PREDICTION
In this section, we will use a beam prediction algorithm to
validate our dataset.
We consider a mobile cellular network including one base
station (BS) and one moving user equipment (UE). The UE is
communicating with the BS, and both line-of-sight (LOS) and
none-line-of-sight (NLOS) exist during the movement. Since
the future networks are likely to coexist in sub-6 GHz and
mmWave bands, we assume that the BS is equipped with two
antenna arrays. One works at the sub-6 GHz band and the
other works at the mmWave band.
In our method, only the sub-6 GHz uplink (UL) channels
is utilized for beam prediction. During the UL signaling, UE
sends pilot signals to the BS in each scheduling time frame,
and the BS receives the UL signal. Denote yup[k] as the
received UL signal at the k-th subcarrier, yup[k] can be shown
as
yup[k] = hup[k]sp + nup[k], (8)
where hup[k] denotes the UL sub-6 GHz channel and sp
denotes the signal transmitted from UE. nup[k] represents the
additive white Gaussian noise (AWGN).
Let hdown[k] denotes the DL channel. The received signal
of the UE at both sub-6 GHz and mmWave bands is given by
ydown[k] = hdown[k]fsd + ndown[k], (9)
where sd represents the signal transmitted from the BS and
ndown[k] represents the AWGN. For the sub-6 GHz band, f
TABLE I
COMPARISON BETWEEN DATASETS
Dataset DeepMIMO Wireless AI Research Dataset DataAI-6G
Multi-band X X X
Spatial non-stationary features X
Number of antenna configurations X X X
Antenna rotation and pattern X X X
Arbitrary multi-user point X
Selective activation of BS and users X X X
User moving function with Doppler X
Customized moving path and speed X
BW Customization X X X
denotes the sub-6 GH beamforming (BF) vectors which can
be obtained by matched filtering. f sub6 can be written as
f sub6 =
h
up[k]
|hup[k]|
. (10)
In the millimeter wave band, a large number of antennas
will be used, resulting in a high overhead using the direct
calculation method. Therefore, in order to reduce the overhead
in high-band communications, we generally use codebooks for
beam selection, fmmW ∈ FmmW which denotes the mmWave
BF vectors. FmmW is a set of pre-prepared beamforming
vectors. Denote P
2 as the DL transmit signal-to-noise ratio
(SNR). The DL data rate for both sub-6 GHz and mmWave
channels can be shown as
R(hdown[k], f ) = Blog2(1 +
P
2
hdown[k]f
2
). (11)
The optimal mmWave BF vector f  is selected to maximize
the mmWave rates R. And the optimal BF vector f  is utilized
to train the machine learning model. f  can be given by
f  = argmax
fmmW2FmmW
R(hdown[k], fmmW). (12)
In this method, we will use the model in [5] and the DataAI-
6G dataset to predict the DL optimal beam at 60 GHz at time
slot t+1 using the UL channel response at 3.5 GHz from time
slot t-24 to time slot t. To select the optimal beam of 60 GHz
for training and testing, an N-phase codebook C is utilized.
Each code in C can be utilized to generate a beam fmmW, and
all beams form a beam set FmmW. The method of selecting
the optimal beam from FmmW is shown in Eq.(12).
We choose TX3 BS and RX6 UE area, and set the user
moving in this area at the speed of 72 km/h, 90 km/h and
108 km/h. The user moves in the positive direction of the xaxis
with the sample frequency of 1 kHz. In order to be able
to compare with the dataset in [5], we take more than 220k
points in total, making the volume of the data comparable to
that in [5].
More detail, we choose the dataset of 3.5 GHz to generate
the UL channel response and set the number of base station
antennas to 16. The dataset of 60 GHz is used to generate the
DL channel response, and the number of base station antennas
is set to 64. The UE is equipped with an omnidirectional
antenna. These settings are consistent with those in [5]. The
BW of UL and DL channel are both set to 100 MHz with
different carrier frequencies, and antenna pattern of UE and
BS are set to isotropic.
In the model training phase, we used the same LSTM model
as in [5]. In addition, to further improve the prediction accuracy,
we combined the LSTM model with convolutional neural
network, thus improving the feature extraction capability of
the model. The structure and parameters of the Conv-LSTM
model are shown in Fig. 5 and Table 2.
Fig. 5. Convolutional layer + LSTM model.
TABLE II
HYPER-PARAMETERS OF THE DESIGNEDMODEL
Parameter Beam
Solver Adam
Activation tanh(LSTM), relu(Dense)
Batch size 32
Max. number of epochs 250
Learning rate 0.0005
LFC(MFC) 2(64,256)
LLSTM(MLSTM) 2(64,128)
Conv2d(filters, size) 32, 1×5
Maxpooling(size) 1×2
Dataset split 80%-20%
The accuracy of correctly predicting the optimal beam
is used as the evaluation criterion. The results of different
datasets is shown in Fig. 6. The first column of the results
is obtained by training the LSTM model with Wireless Insite
data, and the accuracy has reached 88.20% in [5]. As users
in Wireless Insite are not moving, we assume the accuracy
can reach 88.20% at all speed. The second column of the
results is obtained by training the LSTM model with DataAI-
6G dataset. The accuracy of this case has reached 91.65%,
91.03% and 88.85% at the speed of 72 km/h, 90 km/h and 108
km/h. Comparing these two cases, we can find that model has
higher accuracy using DataAI-6G dataset, which means that
our dataset provides more realistic channel features and is well
adapted to the existing algorithms.
72 90 108
Speed(km/h)
0
0.2
0.4
0.6
0.8
1
1.2
Accuracy
LSTM model with Wireless Insite data
LSTM model with DataAI-6G dataset
Fig. 6. Beam prediction accuracy with Wireless Insite and DataAI-6G vs.
Speed of movement.
Then, with the same use of the DataAI-6G dataset comparing
the accuracy of two different models, Conv-LSTM
model has a better performance than LSTM model. Further
observation of the transformation of accuracy with speed, as
shown in Fig. 7, we can find that (i) the accuracy decreases
with increasing speed, which indicates that the difficulty of
beam prediction increases with speed. This is consistent with
reality and reflects a high degree of realism in the mobile
features of our dataset, (ii) the accuracy decreases at a more
moderate rate when using Conv-LSTM model, which indicates
that Conv-LSTM model has a better adaptation to speed.
Therefore, borrowing algorithms from computer vision into
algorithms for channel prediction is a direction that can be
investigated in the future.
V. CONCLUSION
Combining AI with wireless communication is a promising
development direction for future 6G mobile communication.
To meet the future research, the DataAI-6G dataset takes into
account the Doppler properties and, based on this, we implement
a fully user-defined move function and interpolation
function for the first time. Our dataset also considers spatial
non-stationary properties, which are not considered in most
other datasets. In the future, our dataset will further consider
the communication containing RIS. And, in order to meet more
research, we will also add environment and material data in
future iterations of the dataset.
72 90 108
Speed(km/h)
0.8
0.82
0.84
0.86
0.88
0.9
0.92
0.94
0.96
0.98
1
Accuracy
LSTM model with DataAI-6G
Conv-LSTM model with DataAI-6G
Fig. 7. Beam prediction accuracy with LSTM model and Conv-LSTM model
vs. Speed of movement.
REFERENCES
[1] H. Tataria et al., “6G Wireless Systems: Vision, Requirements, Challenges,
Insights, and Opportunities,” in Proceedings of the IEEE, vol.
109, no. 7, pp. 1166-1199, July 2021.
[2] C. Huang et al., “Artificial Intelligence Enabled Radio Propagation for
Communications—Part II: Scenario Identification and Channel Modeling,”
in IEEE Transactions on Antennas and Propagation, vol. 70, no.
6, pp. 3955-3969, June 2022.
[3] Z. Zhang et al., “AI-Based Time-, Frequency-, and Space-Domain
Channel Extrapolation for 6G: Opportunities and Challenges,” in IEEE
Vehicular Technology Magazine, vol. 18, no. 1, pp. 29-39, March 2023.
[4] J. Guo et al., “Convolutional Neural Network-Based Multiple-Rate Compressive
Sensing for Massive MIMO CSI Feedback: Design, Simulation,
and Analysis,” in IEEE Transactions on Wireless Communications, vol.
19, no. 4, pp. 2827-2840, April 2020.
[5] X. Li et al., “Diffraction Characteristics Aided Blockage and Beam
Prediction for mmWave Communications,” 2022 IEEE 95th Vehicular
Technology Conference: (VTC2022-Spring), Helsinki, Finland, 2022,
pp. 1-5.
[6] L. Yu et al., “Long-Range Blockage Prediction Based on Diffraction
Fringe Characteristics for mmWave Communications,” in IEEE Communications
Letters, vol. 26, no. 7, pp. 1683-1687, July 2022.
[7] Y. Sun et al., “Environment Features-Based Model for Path Loss
Prediction,” in IEEE Wireless Communications Letters, vol. 11, no. 9,
pp. 2010-2014, Sept. 2022.
[8] Remcom, “Wireless insite,” http://www.remcom.com/wireless-insite.
[9] W. Khawaja et al., “Indoor Coverage Enhancement for mmwave Systems
with Passive Reflectors: Measurements and Ray Tracing Simulations,”
arXiv preprint arXiv:1808.06223, 2018.
[10] Q. Li et al., “Validation of a Geometry-based Statistical mmwave
Channel Model Using Ray-tracing Simulation,” in 2015 IEEE 81st V
ehicular Technology Conference (VTC Spring), May 2015, pp. 1–5.
[11] S. Wu et al., “Intra-cluster Characteristics of 28 Ghz Wireless Channel
in Urban Micro Street Canyon,” in 2016 IEEE Global Communications
Conference (GLOBECOM), Dec 2016, pp. 1–6.
[12] Z. Yuan et al, “Spatial Non-Stationary Near-Field Channel Modeling
and Validation for Massive MIMO Systems,” in IEEE Transactions on
Antennas and Propagation, vol. 71, no. 1, pp. 921-933, Jan. 2023.
[13] Y. Yang et al., “Deep Learning-Based Downlink Channel Prediction for
FDD Massive MIMO System,” in IEEE Communications Letters, vol.
23, no. 11, pp. 1994-1998, Nov. 2019.
[14] DeepMIMO Dataset. [Online]. Available: http://www.DeepMIMO.net
[15] Wireless AI Research Dataset. [Online]. Available: https://www.mobileai
-dataset.com/html/default/zhongwen/shujuji/1592719963402108929.html
?index=1
