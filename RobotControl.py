import math
import numpy as np
from dynamixel_sdk import *                 # Uses Dynamixel SDK libra
from time import sleep

# Control table address
ADDR_PRO_BAUDRATE           = 8                 # Control table address is different in Dynamixel model

# Protocol version
PROTOCOL_VERSION            = 2.0               # See which protocol version is used in the Dynamixel

# Default setting
DXL_ID0                      = 0                # Dynamixel ID : 0
DXL_ID1                      = 1                # Dynamixel ID : 1
BAUDRATE                    = 57600             # Dynamixel default baudrate : 57600
DEVICENAME                  = 'COM3'    # Check which port is being used on your controller
                                                # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"

FACTORYRST_DEFAULTBAUDRATE  = 57600             # Dynamixel baudrate set by factoryreset
NEW_BAUDNUM                 = 1                 # New baudnum to recover Dynamixel baudrate as it was
OPERATION_MODE              = 0x01              # 0xFF : reset all values
                                                # 0x01 : reset all values except ID
                                                # 0x02 : reset all values except ID and baudrate
TORQUE_ENABLE               = 1                 # Value for enabling the torque
TORQUE_DISABLE              = 0                 # Value for disabling the torque
ADDR_PRO_TORQUE_ENABLE      = 64               # Control table address is different in Dynamixel model
ADDR_PRO_GOAL_POSITION      = 116
ADDR_PRO_PRESENT_POSITION   = 132

P_VALUE                     = 1
I_VALUE                     = 1
D_VALUE                     = 1

MOTOR1 = 1
MOTOR2 = 2

nQ = 1          # Q parameters total
nS = 1          # Number of segments
nQpS = int(nQ/nS)    # Q parameters per segment


def arc2x (kappa, phi, length):
    #Arc2x takes configuration space variables and converts to task space.
    # Kappa is the curveature of each segment
    # Phi is the rotation of each segment. 
    # Length is the length of the back bone of each segment.
    # Returns a task space matrix.
    initFrame = np.identity(4)
    endFrame = initFrame

    for i in range(n):
        # c means current  
        ckappa = kappa[i]
        cphi = phi[i]
        clength = length[i]
        
        #Temp terms
        s1 = np.asarray([[math.cos(cphi), -math.sin(cphi), 0, 0],
              [math.sin(cphi), math.cos(cphi), 0, 0], 
              [0, 0, 1, 0], 
              [0, 0, 0, 1]])
        s2 = 0
        if ckappa == 0: #If straight
            s2 = np.asarray([[0, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, clength],
                 [0, 0, 0, 1]])
        elif ckappa != 0:
            r = 1 / ckappa
            t = ckappa * clength
            s2 = np.asarray([[math.cos(t), 0, math.sin(t), r * (1-math.cos(t))],
                  [0, 1, 0, 0],
                  [-math.sin(t), 0, math.cos(t), r*math.sin(t)], 
                  [0, 0, 0, 1]])

        currentFrame = np.matmul(s1, s2)

        endFrame = np.matmul(endFrame, currentFrame)
    return endFrame

def matrixLog(mat): 
    rot = mat[:3, :3]
    pos = mat[:3,3]

    trace = np.trace(rot)
    theata = math.acos((trace-1)/2)
    cutoff = 3-(1e-4) #problem with float, this fixes it

    w = np.zeros((3,3))
    v=0
    #Special Case
    if(trace >= cutoff):
        v = np.linalg.norm(pos) * pos
        theata = np.linalg.norm(pos)
    # Normal case
    else:
        w = (1/(2*math.sin(theata))) * (rot-rot.transpose())
        #Creating G in parts
        p1 = (1/theata) * np.identity(3)
        p2 = 0.5*w
        p3 = (1/theata) - (0.5/math.tan(theata/2))
        p4 = (w * w)
        g = p1 - p2 + (p3*p4)
        v = np.matmul(g, pos)

    newMatrix = np.zeros((4,4))
    newMatrix[:3, :3] = w
    newMatrix[:3, 3] = v

    return newMatrix

def bodyTwist(current, target):
    # Body twist calculates the difference between two points in task space. 
    tbd = np.matmul(np.linalg.inv(current), target)

    logged = matrixLog(tbd)

    wx = logged[2, 1]
    wy = logged[0,2]
    wz = logged[1,0]
    x  = logged[0,3]
    y  = logged[1,3]
    z  = logged[2,3]

    twist = [wx, wy, wz, x, y, z]
    return twist

def fkine(qN):
    # Converts qoint space parameters to task space.
    q = np.copy(qN)
    backbone = 345 #mm
    l= backbone
    diskrad = 47/2 #mm
    initFrame = np.identity(4)

    # Consistent Variables
    kappa = [0, 0]
    phi = [0, 0]
    length = [l, l]

    # SEgment Variables
    kappaA = 0
    kappaX = 0
    kappaY = 0
    phiA = 0
    q[0] = q[0] - 25
    kappa = [q[0]/(l*diskrad)]
    phi = [0]
    #for i in range(nS):
    #    for j in range(nQpS):
    #        kappaC = 0
    #        if(q[j] != 0):
    #            kappaC = q[j]/(l*diskrad)
#
   #         if(j == 0):
   #             kappaX = kappaC
   #         else:
   #             kappaY = kappaC
#
    #    #kappaA = math.sqrt(math.pow(kappaX, 2) + math.pow(kappaY, 2))
   #     kappaA = kappaX
    #    phiA = 0
    #    #phiA = math.atan2(kappaY, kappaX)

    #    kappa[i] = kappaA
    #    phi[i] = phiA
    location = arc2x(kappa, phi, length, 1, True)
    #x = location[0,3]
    return location


def bodyJacobian(q, n):
    #Jacobian Variables
    jacobian = np.zeros((6, nQ))
    initq = q
    oldFrame = fkine(q)
    h = 1e-1

    #Jacobian Creation
    for i in range(nQ):
        newQ = np.copy(q)
        newQ[i] = q[i] +h

        newFrame =  fkine(newQ)
        twist = bodyTwist(newFrame, oldFrame)

        column = np.transpose(twist)

        jacobian[:, i] = column

    return jacobian

def getError(targetFrame, currentFrame):
    posT = targetFrame[2:4,4]
    posC = currentFrame[2:4,4]

    posErr = posC-posT # This may be the other way around who knows. Maybe change this later in the future.

def inverseKine(targetFrame, current, jacobian, maxError, gainP, gainI=0, prevTwist=[0,0,0,0,0,0]):
    q_target = np.zeros((nQ, 1))

    #gainP = 0.00001
    twist = bodyTwist(current, targetFrame)
    for i in range(6):
        prevTwist[i] += twist[i]
        twist[i] = gainP*twist[i] + prevTwist[i]*gainI
    p1 = np.matmul(np.transpose(jacobian), jacobian)
    p2 = 1/p1
    p3 = p2 * np.transpose(jacobian)
    inverseJac = np.copy(p3)
    #inverseJac = np.transpose(jacobian)*(jacobian*np.linalg.inv(np.transpose(jacobian)))
    q_change = np.matmul(inverseJac, twist)

    return q_change, prevTwist

def readMotorPosition(id):
    #dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRO_PRESENT_POSITION)
    dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, id, ADDR_PRO_PRESENT_POSITION)
    return dxl_present_position


def torqueEnable(id):
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, id, ADDR_PRO_TORQUE_ENABLE, TORQUE_ENABLE)


def writeMotorPosition(id, pos):
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, id, ADDR_PRO_GOAL_POSITION, pos)


def openP():
    # Initialize PortHandler instance
    # Set the port path
    # Get methods and members of PortHandlerLinux or PortHandlerWindows

    # Try factoryreset
    print("[ID:%03d] Try factoryreset : " % DXL_ID0)
    print("[ID:%03d] Try factoryreset : " % DXL_ID1)

    dxl_comm_result1, dxl_error1 = packetHandler.factoryReset(portHandler, DXL_ID0, OPERATION_MODE)
    dxl_comm_result2, dxl_error2 = packetHandler.factoryReset(portHandler, DXL_ID1, OPERATION_MODE)
    if dxl_comm_result1 != COMM_SUCCESS:
        print("Aborted")
        quit()
    if dxl_comm_result2 != COMM_SUCCESS:
        print("Aborted")
        quit()

    # Wait for reset
    print("Wait for reset...")
    sleep(2.0)
    print("[ID:%03d] factoryReset Success!" % DXL_ID0)
    print("[ID:%03d] factoryReset Success!" % DXL_ID1)

def motor2Q(motor):
    x = abs(motor-4095)
    deg = x/(4095/360) # To get degrees.
    rad = deg * 0.01745
    length = rad * (8.5/2) # 8.5 is average diameter of the screw.
    q = length

    return q

def q2motor(q):
    rad = q/(8.5/2)
    deg = rad  / 0.01745
    x = deg*(4095/360)
    motor = int(abs(4095-x))
    return motor


def moveTendons(q):
    for i in range(nQ):
        qi = q[i]
        if qi > 26:
            qi = 26
        if qi < 0:
            qi = 0
        q[i] = qi
        motorDisplacement = q2motor(qi)
        writeMotorPosition(i, motorDisplacement)
