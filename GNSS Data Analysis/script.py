import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import math
import pandas as pd
from mpl_toolkits import mplot3d


# Read csv data into pandas Dataframe
def read_data(file):
  data = pd.read_csv(file)
  return data

# Convert ECEF (m) coordinates to ENU (m) about reference latitude (N°) and longitude (E°)
def ecef2enu(x,y,z, lat_ref, lon_ref):
  lat_ref = np.deg2rad(lat_ref)
  lon_ref = np.deg2rad(lon_ref + 360)
  C = np.zeros((3,3))
  C[0,0]=-np.sin(lat_ref)*np.cos(lon_ref)
  C[0,1]=-np.sin(lat_ref)*np.sin(lon_ref)
  C[0,2]= np.cos(lat_ref)

  C[1,0]=-np.sin(lon_ref)
  C[1,1]= np.cos(lon_ref)
  C[1,2]= 0

  C[2,0]=np.cos(lat_ref)*np.cos(lon_ref)
  C[2,1]=np.cos(lat_ref)*np.sin(lon_ref)
  C[2,2]=np.sin(lat_ref)

  x, y, z = np.dot(C,np.array([x, y, z]))

  return x, y, z


# Convert zero-centered Cartesian coordinates to Elevation (°) and Azimuth (°)
def cart2elaz(x,y,z):
    R = x**2 + y**2
    elev = np.rad2deg(math.atan2(z,math.sqrt(R)))
    az = np.rad2deg(math.atan2(y,x))
    return elev, az

# Create an empty skyplot with a closure to add satellites
def create_skyplot():
  ax = plt.gca(projection='polar')
  ax.set_theta_zero_location('N')
  ax.set_theta_direction(-1)
  ax.set_rlim(1, 91)
  ax.grid(True, which='major')
  degree_sign = u'\N{DEGREE SIGN}'
  r_labels = [
      '90' + degree_sign,
      '',
      '60' + degree_sign,
      '',
      '30' + degree_sign,
      '',
      '0' + degree_sign,
  ]
  ax.set_rgrids(range(1, 106, 15), r_labels, angle=-45)

  # Closure to add satellites to the skyplot from list of Elevation (°) and Azimuth (°)
  def add_satellite(trajectory, label=None): 
    trajectory = np.array(trajectory)
    ax.scatter(trajectory[:, 1], trajectory[:, 0], marker='x', label=label)
  
  return add_satellite, ax

# Utility function to compute expected pseudorange from (x, y, z, cdt) to satellite (X, Y, Z, B)
def expected_prange(x, y, z, cdt, X, Y, Z, B):
  return np.sqrt((x - X)**2 + (y - Y)**2 + (z - Z)**2) + cdt - B

# Pseudoinverse of matrix A using diagonal weight matrix W
def weighted_pinv(A, W=None):
    if W is None:
        W = np.diag(np.ones(len(A)))
    _A = np.linalg.inv(A.T @ W @ A)
    return _A @ A.T @ W

##########################################################################################################################
##                          Main Code                                                                                   ##
data = read_data('hw1_data.csv')


######################################################################
# 2.1 
def plot_sats(data):
  satIDs = np.unique((data['svid']).to_numpy()) # Find all satellites. Different satellites in each time epoch. Some never visible due to NLOS
  nOfSats = len(satIDs)
  add_satellite_All, skyplotAll = create_skyplot() # New plot
  for i in range(0, nOfSats): # iterate for every satellite: find the satellite's trajectory 
    satData = data.loc[data['svid']==satIDs[i]]
    satTraj = satData[['xSatPosM', 'ySatPosM', 'zSatPosM']].to_numpy()
    for k in range(0, satTraj.shape[0]):
      x, y, z = ecef2enu(satTraj[k,0], satTraj[k,1], satTraj[k,2], 37.3715, -122.047861)
      satTraj[k, 0] = x
      satTraj[k, 1] = y
      satTraj[k, 2] = z
  
    satTrajAlEl = np.zeros((satTraj.shape[0], 2))
    for j in range(0, satTraj.shape[0]):
      satTrajAlEl[j,:] = cart2elaz(satTraj[j, 0], satTraj[j, 1], satTraj[j,2])

    l = "{}".format(satIDs[i])
    add_satellite_All(satTrajAlEl, satIDs[i])
  plt.legend(bbox_to_anchor=(0, 1), loc='upper right', ncol=1)
  plt.show()

epochs = np.sort(np.unique(data['millisSinceGpsEpoch'].to_numpy())) # Find all epochs
firstEpoch = np.min(epochs) # Find first epoch
firstEpochData = data.loc[data['millisSinceGpsEpoch']==firstEpoch] # Get the data of the first epoch

plot_sats(firstEpochData)

######################################################################
# 2.2
plot_sats(data)

######################################################################
# 2.3 
# No code needed. Compare the 2 skyplots

######################################################################
# 2.4
def newtonRaphson_ECEF(data, initialEstimate, kmax, epoch, nSats=0):
  newEstimate = np.zeros((4,1)) 
  newEstimate[0,0] = initialEstimate[0,0]
  newEstimate[1,0] = initialEstimate[1,0]
  newEstimate[2,0] = initialEstimate[2,0]
  newEstimate[3,0] = initialEstimate[3,0]

  epochData = data.loc[data['millisSinceGpsEpoch']==epoch]
  s = np.unique((epochData['svid']).to_numpy())

  if nSats !=0:
    s = np.random.choice(s, nSats, replace=False)

  for k in range(0, kmax):
    F = np.zeros((len(s),1))
    A = np.zeros((len(s),4))
    j = 0
    for i in s:
      satData = epochData.loc[data['svid']==i]

      r_i = satData['PrM']
      b_i = satData['satClkBiasM']
      x_sat = satData['xSatPosM']
      y_sat = satData['ySatPosM']
      z_sat = satData['zSatPosM']

      F[j,0] = r_i - (math.sqrt(math.pow((newEstimate[0]-x_sat),2) + math.pow((newEstimate[1]-y_sat),2) + math.pow((newEstimate[2]-z_sat),2)) + newEstimate[3] - b_i)
      A[j,0] = - (newEstimate[0]-x_sat)/math.sqrt(math.pow((newEstimate[0]-x_sat),2) + math.pow((newEstimate[1]-y_sat),2) + math.pow((newEstimate[2]-z_sat),2))
      A[j,1] = - (newEstimate[1]-y_sat)/math.sqrt(math.pow((newEstimate[0]-x_sat),2) + math.pow((newEstimate[1]-y_sat),2) + math.pow((newEstimate[2]-z_sat),2))
      A[j,2] = - (newEstimate[2]-z_sat)/math.sqrt(math.pow((newEstimate[0]-x_sat),2) + math.pow((newEstimate[1]-y_sat),2) + math.pow((newEstimate[2]-z_sat),2))
      A[j,3] = -1
      
      j = j+1
    
    pseudoInv_A = weighted_pinv(A)
    newEstimate = newEstimate + np.matmul(pseudoInv_A,-F)
  return newEstimate

def newtonRaphson_ECEF_AllEpochs(data, epochs, initialEstimate, kmax, nSats=0):
  estimatesAllEpochs_ECEF = np.zeros((4,len(epochs)))
  k = 0
  for i in epochs:
      result = newtonRaphson_ECEF(data, initialEstimate, kmax, i, nSats)
      estimatesAllEpochs_ECEF[0,k] = result[0,0]
      estimatesAllEpochs_ECEF[1,k] = result[1,0]
      estimatesAllEpochs_ECEF[2,k] = result[2,0]
      estimatesAllEpochs_ECEF[3,k] = result[3,0]
      k = k+1
  return estimatesAllEpochs_ECEF

# Calculate Newton Raphson Estimate in ECEF, with All Satellites and Initial Position as Initial Estimate
print("ECEF Estimate with All Satellites - Initial Position as Initial Estimate")
estimatesAllEpochs_InitialPosEst_ECEF = newtonRaphson_ECEF_AllEpochs(data, epochs, np.array([[-2692974], [-4301659], [3850240],[0]]), 5, 0)
print(estimatesAllEpochs_InitialPosEst_ECEF[:,0])

# Calculate Newton Raphson Estimate in ECEF, with All Satellites and Initial Position as Initial Estimate
print("ECEF Estimate with All Satellites - Zero as Initial Estimate")
estimatesAllEpochs_InitialZeroEst_ECEF = newtonRaphson_ECEF_AllEpochs(data, epochs, np.array([[0], [0], [0],[0]]), 5, 0)
print(estimatesAllEpochs_InitialZeroEst_ECEF[:,0])

# Plots
plt.plot(epochs, estimatesAllEpochs_InitialPosEst_ECEF[0,:], label = 'Initial Position as Initial Estimate')
plt.plot(epochs, estimatesAllEpochs_InitialZeroEst_ECEF[0,:], label = 'Zero as Initial Estimate')
plt.xlabel('Epochs')
plt.ylabel('X Coordinate Estimate [m]')
plt.title('X Coordinate Estimate')
plt.legend()
plt.show()

plt.plot(epochs, estimatesAllEpochs_InitialPosEst_ECEF[1,:], label = 'Initial Position as Initial Estimate')
plt.plot(epochs, estimatesAllEpochs_InitialZeroEst_ECEF[1,:], label = 'Zero as Initial Estimate')
plt.xlabel('Epochs')
plt.ylabel('Y Coordinate Estimate [m]')
plt.title('Y Coordinate Estimate')
plt.legend()
plt.show()

plt.plot(epochs, estimatesAllEpochs_InitialPosEst_ECEF[2,:], label = 'Initial Position as Initial Estimate')
plt.plot(epochs, estimatesAllEpochs_InitialZeroEst_ECEF[2,:], label = 'Zero as Initial Estimate')
plt.xlabel('Epochs')
plt.ylabel('Z Coordinate Estimate [m]')
plt.title('Z Coordinate Estimate')
plt.legend()
plt.show()

plt.plot(epochs, estimatesAllEpochs_InitialPosEst_ECEF[3,:], label = 'Initial Position as Initial Estimate')
plt.plot(epochs, estimatesAllEpochs_InitialZeroEst_ECEF[3,:], label = 'Zero as Initial Estimate')
plt.xlabel('Epochs')
plt.ylabel('Clock Bias Multipled by Speed of Light')
plt.title('Clock Bias')
plt.legend()
plt.show()

######################################################################
# 2.5
def newtonRaphson_ENU(data, initialEstimate, kmax, epoch):
  newEstimate = np.zeros((4,1)) 
  newEstimate[0,0] = initialEstimate[0,0]
  newEstimate[1,0] = initialEstimate[1,0]
  newEstimate[2,0] = initialEstimate[2,0]
  newEstimate[3,0] = initialEstimate[3,0]

  epochData = data.loc[data['millisSinceGpsEpoch']==epoch]
  s = np.unique((epochData['svid']).to_numpy())

  for k in range(0, kmax):
    F = np.zeros((len(s),1))
    A = np.zeros((len(s),4))
    j = 0
    for i in s:
      satData = epochData.loc[data['svid']==i]

      r_i = satData['PrM']
      b_i = satData['satClkBiasM']
      x_sat, y_sat, z_sat = ecef2enu(satData['xSatPosM'],satData['ySatPosM'],satData['zSatPosM'], 37.3715, -122.047861)

      F[j,0] = r_i - (math.sqrt(math.pow((newEstimate[0]-x_sat),2) + math.pow((newEstimate[1]-y_sat),2) + math.pow((newEstimate[2]-z_sat),2)) + newEstimate[3] - b_i)
      A[j,0] = - (newEstimate[0]-x_sat)/math.sqrt(math.pow((newEstimate[0]-x_sat),2) + math.pow((newEstimate[1]-y_sat),2) + math.pow((newEstimate[2]-z_sat),2))
      A[j,1] = - (newEstimate[1]-y_sat)/math.sqrt(math.pow((newEstimate[0]-x_sat),2) + math.pow((newEstimate[1]-y_sat),2) + math.pow((newEstimate[2]-z_sat),2))
      A[j,2] = - (newEstimate[2]-z_sat)/math.sqrt(math.pow((newEstimate[0]-x_sat),2) + math.pow((newEstimate[1]-y_sat),2) + math.pow((newEstimate[2]-z_sat),2))
      A[j,3] = -1
      
      j = j+1
    
    pseudoInv_A = weighted_pinv(A)
    newEstimate = newEstimate + np.matmul(pseudoInv_A,-F)
  return newEstimate

def newtonRaphson_ENU_AllEpochs(data, epochs, initialEstimate, kmax, nSats=0):
  estimatesAllEpochs_ENU = np.zeros((4,len(epochs)))
  k = 0
  for i in epochs:
    result = newtonRaphson_ENU(data, initialEstimate, kmax, i)
    estimatesAllEpochs_ENU[0,k] = result[0,0]
    estimatesAllEpochs_ENU[1,k] = result[1,0]
    estimatesAllEpochs_ENU[2,k] = result[2,0]
    estimatesAllEpochs_ENU[3,k] = result[3,0]
    k = k+1

  return estimatesAllEpochs_ENU

# Determine ENU Solution from Newton-Raphson
print("ENU Estimate with All Satellites")
estimatesAllEpochs_ENU = newtonRaphson_ENU_AllEpochs(data, epochs, np.array([[-20621.87590879528], [-7.501155948266387], [6370267.020520784], [0.0]]), 5, 0)
print(estimatesAllEpochs_ENU[:,0])
print("ECEF Estimate with All Satellites Transformed to ENU")
print(ecef2enu(estimatesAllEpochs_InitialPosEst_ECEF[0,0], estimatesAllEpochs_InitialPosEst_ECEF[1,0], estimatesAllEpochs_InitialPosEst_ECEF[2,0], 37.3715, -122.047861))

# Convert ECEF Solution with All Satellites to ENU
estimatesAllEpochs_InitialPosEst_ECEFtoENU = np.zeros((4, len(epochs)))
k = 0
for i in epochs:
  estimatesAllEpochs_InitialPosEst_ECEFtoENU[0,k], estimatesAllEpochs_InitialPosEst_ECEFtoENU[1,k], estimatesAllEpochs_InitialPosEst_ECEFtoENU[2,k] = ecef2enu(estimatesAllEpochs_InitialPosEst_ECEF[0,k], estimatesAllEpochs_InitialPosEst_ECEF[1,k], estimatesAllEpochs_InitialPosEst_ECEF[2,k], 37.3715, -122.047861)
  estimatesAllEpochs_InitialPosEst_ECEFtoENU[3,k] = estimatesAllEpochs_InitialPosEst_ECEF[3, k]
  k = k+1

# Plots
plt.plot(epochs, estimatesAllEpochs_ENU[0,:], label = 'ENU Estimate')
plt.plot(epochs, estimatesAllEpochs_InitialPosEst_ECEFtoENU[0,:], label = 'ECEF Estimate as ENU')
plt.xlabel('Epochs')
plt.ylabel('X Coordinate Estimate [m]')
plt.title('X Coordinate Estimate')
plt.legend()
plt.show()

plt.plot(epochs, estimatesAllEpochs_ENU[1,:], label = 'ENU Estimate')
plt.plot(epochs, estimatesAllEpochs_InitialPosEst_ECEFtoENU[1,:], label = 'ECEF Estimate as ENU')
plt.xlabel('Epochs')
plt.ylabel('Y Coordinate Estimate [m]')
plt.title('Y Coordinate Estimate')
plt.legend()
plt.show()

plt.plot(epochs, estimatesAllEpochs_ENU[2,:], label = 'ENU Estimate')
plt.plot(epochs, estimatesAllEpochs_InitialPosEst_ECEFtoENU[2,:], label = 'ECEF Estimate as ENU')
plt.xlabel('Epochs')
plt.ylabel('Z Coordinate Estimate [m]')
plt.title('Z Coordinate Estimate')
plt.legend()
plt.show()

plt.plot(epochs, estimatesAllEpochs_ENU[3,:], label = 'ENU Estimate')
plt.plot(epochs, estimatesAllEpochs_InitialPosEst_ECEFtoENU[3,:], label = 'ECEF Estimate as ENU')
plt.xlabel('Epochs')
plt.ylabel('Clock Bias Mutlipled by Speed of Light')
plt.title('Clock Bias')
plt.legend()
plt.show()

######################################################################
# 2.6

# Calculate Newton Raphson estimate using limited number of satellites. The choice of satellites is independent in each epoch.
estimate_ECEF_4 = newtonRaphson_ECEF_AllEpochs(data, epochs, np.array([[-2692974], [-4301659], [3850240],[0]]), 5, 4)
estimate_ECEF_5 = newtonRaphson_ECEF_AllEpochs(data, epochs, np.array([[-2692974], [-4301659], [3850240],[0]]), 5, 5)
estimate_ECEF_6 = newtonRaphson_ECEF_AllEpochs(data, epochs, np.array([[-2692974], [-4301659], [3850240],[0]]), 5, 6) 
print("ECEF Estimate with 4 Satellites")
print(estimate_ECEF_4[:,0])
print("ECEF Estimate with 5 Satellites")
print(estimate_ECEF_5[:,0])
print("ECEF Estimate with 6 Satellites")
print(estimate_ECEF_6[:,0]) 
  # As number of sats increases, the estimate closer to our expected position

plt.plot(epochs, estimate_ECEF_4[0,:], label = '4 Satellites')
plt.plot(epochs, estimate_ECEF_5[0,:], label = '5 Satellites')
plt.plot(epochs, estimate_ECEF_6[0,:], label = '6 Satellites')
plt.plot(epochs, estimatesAllEpochs_InitialPosEst_ECEF[0,:], label = 'ECEF Solution - All Satellites')
plt.xlabel('Epochs')
plt.ylabel('X Coordinate Estimate [m]')
plt.title('X Coordinate Estimate')
plt.legend()
plt.show()

plt.plot(epochs, estimate_ECEF_4[1,:], label = '4 Satellites')
plt.plot(epochs, estimate_ECEF_5[1,:], label = '5 Satellites')
plt.plot(epochs, estimate_ECEF_6[1,:], label = '6 Satellites')
plt.plot(epochs, estimatesAllEpochs_InitialPosEst_ECEF[1,:], label = 'ECEF Solution - All Satellites')
plt.xlabel('Epochs')
plt.ylabel('Y Coordinate Estimate [m]')
plt.title('Y Coordinate Estimate')
plt.legend()
plt.show()

plt.plot(epochs, estimate_ECEF_4[2,:], label = '4 Satellites')
plt.plot(epochs, estimate_ECEF_5[2,:], label = '5 Satellites')
plt.plot(epochs, estimate_ECEF_6[2,:], label = '6 Satellites')
plt.plot(epochs, estimatesAllEpochs_InitialPosEst_ECEF[2,:], label = 'ECEF Solution - All Satellites')
plt.xlabel('Epochs')
plt.ylabel('Z Coordinate Estimate [m]')
plt.title('Z Coordinate Estimate')
plt.legend()
plt.show()

plt.plot(epochs, estimate_ECEF_4[3,:], label = '4 Satellites')
plt.plot(epochs, estimate_ECEF_5[3,:], label = '5 Satellites')
plt.plot(epochs, estimate_ECEF_6[3,:], label = '6 Satellites')
plt.plot(epochs, estimatesAllEpochs_InitialPosEst_ECEF[3,:], label = 'ECEF Solution - All Satellites')
plt.xlabel('Epochs')
plt.ylabel('Clock Bias Multiplied by Speed of Light')
plt.title('Clock Bias')
plt.legend()
plt.show()

######################################################################
# 2.7 

# Calculate Newton Raphson for different groups of 4 satellites
estimate_ECEF_4_1 = newtonRaphson_ECEF_AllEpochs(data, epochs, np.array([[-2692974], [-4301659], [3850240],[0]]), 5, 4)
estimate_ECEF_4_2 = newtonRaphson_ECEF_AllEpochs(data, epochs, np.array([[-2692974], [-4301659], [3850240],[0]]), 5, 4)
estimate_ECEF_4_3 = newtonRaphson_ECEF_AllEpochs(data, epochs, np.array([[-2692974], [-4301659], [3850240],[0]]), 5, 4)

plt.plot(epochs, estimate_ECEF_4_1[0,:], label = '1st Variation')
plt.plot(epochs, estimate_ECEF_4_2[0,:], label = '2nd Variation')
plt.plot(epochs, estimate_ECEF_4_3[0,:], label = '3rd Variation')
plt.xlabel('Epochs')
plt.ylabel('X Coordinate Estimate [m]')
plt.title('X Coordinate Estimate')
plt.legend()
plt.show()

plt.plot(epochs, estimate_ECEF_4_1[1,:], label = '1st Variation')
plt.plot(epochs, estimate_ECEF_4_2[1,:], label = '2nd Variation')
plt.plot(epochs, estimate_ECEF_4_3[1,:], label = '3rd Variation')
plt.xlabel('Epochs')
plt.ylabel('Y Coordinate Estimate [m]')
plt.title('Y Coordinate Estimate')
plt.legend()
plt.show()

plt.plot(epochs, estimate_ECEF_4_1[2,:], label = '1st Variation')
plt.plot(epochs, estimate_ECEF_4_2[2,:], label = '2nd Variation')
plt.plot(epochs, estimate_ECEF_4_3[2,:], label = '3rd Variation')
plt.xlabel('Epochs')
plt.ylabel('Z Coordinate Estimate [m]')
plt.title('Z Coordinate Estimate')
plt.legend()
plt.show()

plt.plot(epochs, estimate_ECEF_4_1[3,:], label = '1st Variation')
plt.plot(epochs, estimate_ECEF_4_2[3,:], label = '2nd Variation')
plt.plot(epochs, estimate_ECEF_4_3[3,:], label = '3rd Variation')
plt.xlabel('Epochs')
plt.ylabel('Clock Bias Multiplied by Speed of Light')
plt.title('Clock Bias')
plt.legend()
plt.show()

######################################################################
# 2.8

residuals = np.zeros((len(np.unique(data['svid'].to_numpy())), len(epochs)))
sats = np.unique(data.svid.to_numpy())
dict = {}

t = 0
for j in sats:
  dict[j] = t
  t = t+1

t = 0
for i in epochs:
  epochData = data.loc[data['millisSinceGpsEpoch']==i]
  s = np.unique((epochData['svid']).to_numpy())
  for j in s:
    satData = epochData.loc[data['svid']==j]

    r_i = satData['PrM']
    b_i = satData['satClkBiasM']
    x_sat = satData['xSatPosM']
    y_sat = satData['ySatPosM']
    z_sat = satData['zSatPosM']

    res = abs(r_i - (math.sqrt(math.pow((estimatesAllEpochs_InitialPosEst_ECEF[0,t]-x_sat),2) + math.pow((estimatesAllEpochs_InitialPosEst_ECEF[1,t]-y_sat),2) + math.pow((estimatesAllEpochs_InitialPosEst_ECEF[2,t]-z_sat),2)) + estimatesAllEpochs_InitialPosEst_ECEF[3,t] - b_i))
    residuals[dict[j], t] = res

  t = t+1

# Plot Residuals
for i in sats:
  plt.plot(epochs, residuals[dict[i],:], label = i)

plt.xlabel('Epochs')
plt.ylabel('Residual [m]')
plt.title('Residuals for Satellite Measurements')
plt.legend()
plt.show()

######################################################################
# 2.9
def wls(data, initialEstimate, kmax, epoch, nSats=0, choice=0):
  newEstimate = np.zeros((4,1)) 
  newEstimate[0,0] = initialEstimate[0,0]
  newEstimate[1,0] = initialEstimate[1,0]
  newEstimate[2,0] = initialEstimate[2,0]
  newEstimate[3,0] = initialEstimate[3,0]

  epochData = data.loc[data['millisSinceGpsEpoch']==epoch]
  s = np.unique((epochData['svid']).to_numpy())

  if nSats !=0:
    s = np.random.choice(s, nSats, replace=False)

  for k in range(0, kmax):
    F = np.zeros((len(s),1))
    A = np.zeros((len(s),4))
    W = np.zeros((len(s),len(s)))
    j = 0
    for i in s:
      satData = epochData.loc[data['svid']==i]

      r_i = satData['PrM']
      b_i = satData['satClkBiasM']
      x_sat = satData['xSatPosM']
      y_sat = satData['ySatPosM']
      z_sat = satData['zSatPosM']
      ionoDelay = satData['ionoDelayM']
      tropoDelay = satData['tropoDelayM']

      F[j,0] = r_i - (math.sqrt(math.pow((newEstimate[0]-x_sat),2) + math.pow((newEstimate[1]-y_sat),2) + math.pow((newEstimate[2]-z_sat),2)) + newEstimate[3] - b_i)
      A[j,0] = - (newEstimate[0]-x_sat)/math.sqrt(math.pow((newEstimate[0]-x_sat),2) + math.pow((newEstimate[1]-y_sat),2) + math.pow((newEstimate[2]-z_sat),2))
      A[j,1] = - (newEstimate[1]-y_sat)/math.sqrt(math.pow((newEstimate[0]-x_sat),2) + math.pow((newEstimate[1]-y_sat),2) + math.pow((newEstimate[2]-z_sat),2))
      A[j,2] = - (newEstimate[2]-z_sat)/math.sqrt(math.pow((newEstimate[0]-x_sat),2) + math.pow((newEstimate[1]-y_sat),2) + math.pow((newEstimate[2]-z_sat),2))
      A[j,3] = -1
      
      if choice == 0:
        W[j,j] = ionoDelay
      else:
        W[j,j] = tropoDelay
      j = j+1

    K1 = np.matmul(np.transpose(A), W)
    K1 = np.matmul(K1, A)
    K1 = np.linalg.inv(K1)
    K2 = np.matmul(np.transpose(A), W)
    K2 = np.matmul(K2, -F)
    newEstimate = newEstimate + np.matmul(K1, K2)
  return newEstimate

def wls_AllEpochs(data, epochs, initialEstimate, kmax, nSats=0, choice=0):
  estimatesAllEpochs_wls = np.zeros((4,len(epochs)))
  k = 0
  for i in epochs:
      result = wls(data, initialEstimate, kmax, i, nSats, choice)
      estimatesAllEpochs_wls[0,k] = result[0,0]
      estimatesAllEpochs_wls[1,k] = result[1,0]
      estimatesAllEpochs_wls[2,k] = result[2,0]
      estimatesAllEpochs_wls[3,k] = result[3,0]
      k = k+1
  return estimatesAllEpochs_wls

# WLS Solution using IonoDelay
estimatesAllEpochs_wls_0 = wls_AllEpochs(data, epochs, np.array([[-2692974], [-4301659], [3850240],[0]]), 5, 0, 0)

# WLS Solution using TropoDelay
estimatesAllEpochs_wls_1 = wls_AllEpochs(data, epochs, np.array([[-2692974], [-4301659], [3850240],[0]]), 5, 0, 1)

plt.plot(epochs, estimatesAllEpochs_wls_0[0,:], label = 'WLS with ionoDelayM')
plt.plot(epochs, estimatesAllEpochs_wls_1[0,:], label = 'WLS with tropoDelayM')
plt.plot(epochs, estimatesAllEpochs_InitialPosEst_ECEF[0,:], label = 'ECEF Solution')
plt.xlabel('Epochs')
plt.ylabel('X Coordinate Estimate [m]')
plt.title('X Coordinate Estimate')
plt.legend()
plt.show()

plt.plot(epochs, estimatesAllEpochs_wls_0[1,:], label = 'WLS with ionoDelayM')
plt.plot(epochs, estimatesAllEpochs_wls_1[1,:], label = 'WLS with tropoDelayM')
plt.plot(epochs, estimatesAllEpochs_InitialPosEst_ECEF[1,:], label = 'ECEF Solution')
plt.xlabel('Epochs')
plt.ylabel('Y Coordinate Estimate [m]')
plt.title('Y Coordinate Estimate')
plt.legend()
plt.show()

plt.plot(epochs, estimatesAllEpochs_wls_0[2,:], label = 'WLS with ionoDelayM')
plt.plot(epochs, estimatesAllEpochs_wls_1[2,:], label = 'WLS with tropoDelayM')
plt.plot(epochs, estimatesAllEpochs_InitialPosEst_ECEF[2,:], label = 'ECEF Solution')
plt.xlabel('Epochs')
plt.ylabel('Z Coordinate Estimate [m]')
plt.title('Z Coordinate Estimate')
plt.legend()
plt.show()

plt.plot(epochs, estimatesAllEpochs_wls_0[3,:], label = 'WLS with ionoDelayM')
plt.plot(epochs, estimatesAllEpochs_wls_1[3,:], label = 'WLS with tropoDelayM')
plt.plot(epochs, estimatesAllEpochs_InitialPosEst_ECEF[3,:], label = 'ECEF Solution')
plt.xlabel('Epochs')
plt.ylabel('Clock Bias Mutliplied by Speed of Light')
plt.title('Clock Bias')
plt.legend()
plt.show()

