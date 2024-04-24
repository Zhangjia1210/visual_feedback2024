import pybullet as p
import numpy as np
import time
from camera import Camera
import cv2

IMG_SIDE = 300
IMG_HALF = IMG_SIDE/2
MARKER_LENGTH = 0.1
MARKER_CORNERS_WORLD = np.array(
    [
        [-MARKER_LENGTH/2,MARKER_LENGTH/2,0.0,1],
        [MARKER_LENGTH/2,MARKER_LENGTH/2,0.0,1],
        [MARKER_LENGTH/2,-MARKER_LENGTH/2.0,0.0,1],
        [-MARKER_LENGTH/2,-MARKER_LENGTH/2,0.0,1]
    ]
)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

def computeInterMatrix(Z, sd0):
    L = np.zeros((8,3))
    for idx in range(4):
        x = sd0[2*idx, 0]
        y = sd0[2*idx+1, 0]
        L[2*idx] = np.array([-1/Z,0,y])
        L[2*idx+1] = np.array([0,-1/Z,-x])
    return L

def updateCamPos(cam):
    linkState = p.getLinkState(boxId, linkIndex=6)
    # pos
    xyz = linkState[0]
    # orientation
    quat = linkState[1]
    rotMat = p.getMatrixFromQuaternion(quat)
    rotMat = np.reshape(np.array(rotMat),(3,3))
    camera.set_new_position(xyz, rotMat)
    

camera = Camera(imgSize = [IMG_SIDE, IMG_SIDE])

dt = 1/240 # pybullet simulation step
q0 = 0.5  # starting position (radian)
qd = 0.5

xd = 0.5
yd = 0.5

L = 0.5
Z0 = 0.3
pos = q0
maxTime = 10
logTime = np.arange(0.0, maxTime, dt)
sz = logTime.size
logPos = np.zeros(sz)
logPos[0] = q0
logVel = np.zeros(sz)

jointIndices = [1,3,5]
eefLinkIdx = 6

#or p.DIRECT for non-graphical version
physicsClient = p.connect(p.GUI, options="--background_color_red=1 --background_color_blue=1 --background_color_green=1")
p.resetDebugVisualizerCamera(
    cameraDistance=0.5,
    cameraYaw=-90,
    cameraPitch=-89.999,
    cameraTargetPosition=[0.5, 0.5, 0.6]
)
p.setGravity(0,0,-10)
boxId = p.loadURDF("combined/simple.urdf.xml", useFixedBase=True)

# add aruco cube and aruco texture
c = p.loadURDF('combined/aruco.urdf', (0.5, 0.5, 0.0), useFixedBase=True)
x = p.loadTexture('combined/aruco_cube.png')
p.changeVisualShape(c, -1, textureUniqueId=x)

numJoints = p.getNumJoints(boxId)
for idx in range(numJoints):
    print(f"{idx} {p.getJointInfo(boxId, idx)[1]} {p.getJointInfo(boxId, idx)[12]}")

print(p.isNumpyEnabled())

# Use Inverse Kinematics tp calculate positions
def move_to_pos(desired_pos, verbose=False):
    jointAngles = p.calculateInverseKinematics(boxId, eefLinkIdx, desired_pos)
    p.setJointMotorControlArray(boxId, jointIndices, p.POSITION_CONTROL, jointAngles)
    for _ in range(100):
        p.stepSimulation()
        if verbose:
            currentPose = p.getLinkState(boxId, eefLinkIdx)[0]
            print(f"Current Position: {currentPose}")

# Initial positions setup...
initial_pos = [(0.2, 0.2, 0.1), (0.3, 0.4, 0.2), (0.5, 0.5, 0.2)]
for position in initial_pos:
    move_to_pos(position, verbose=True)
    print(f"Position reached: {position}")


updateCamPos(camera)
img = camera.get_frame()
corners, markerIds, rejectedCandidates = detector.detectMarkers(img)
sd0 = np.reshape(np.array(corners[0][0]),(8,1))
sd0 = np.array([(s-IMG_HALF)/IMG_HALF for s in sd0])
sd = np.reshape(np.array(corners[0][0]),(8,1)).astype(int)


idx = 1
camCount = 0
w = np.zeros((3,1))
for t in logTime[1:]:
    p.stepSimulation()

    camCount += 1
    if (camCount == 5):
        camCount = 0
        updateCamPos(camera)
        camera.get_frame()
        img = camera.get_frame()
        corners, markerIds, rejectedCandidates = detector.detectMarkers(img)
        s = corners[0][0,0]
        s0 = np.reshape(np.array(corners[0][0]),(8,1))
        s0 = np.array([(ss-IMG_HALF)/IMG_HALF for ss in s0])
        L0 = computeInterMatrix(Z0, s0)
        L0T = np.linalg.inv(L0.T@L0)@L0.T
        e = s0 - sd0
        coef = 1/2
        w = -coef * L0T @ e

    jStates = p.getJointStates(boxId, jointIndices=jointIndices)
    jPos = [state[0] for state in jStates]
    jVel = [state[1] for state in jStates]
    (linJac,angJac) = p.calculateJacobian(
        bodyUniqueId = boxId, 
        linkIndex = 6,
        localPosition = [0,0,0],
        objPositions = jPos,
        objVelocities = [0,0,0],
        objAccelerations = [0,0,0]
    )

    J = np.block([
        [np.array(linJac)[:2,:2], np.zeros((2,1))],
        [np.array(angJac)[2,:]]
    ])
    dq = (np.linalg.inv(J) @ w).flatten()[[1,0,2]]
    dq[2] = -dq[2]
    p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, targetVelocities=dq, controlMode=p.VELOCITY_CONTROL)
    #time.sleep(0.01)
    
p.disconnect()

