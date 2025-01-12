# -*- coding: utf-8 -*-
"""

This code is part of the supplementary materials of the Journal of Glaciology article titled:
 -----------------------------------------------------------------------------------------------------

'Firn densification in two dimensions:
modelling the collapse of snow caves and enhanced densification in ice-stream shear margins'

                            Arrizabalaga-Iriarte J, Lejonagoitia-Garmendia L, Hvidberg CS, Grinsted A, Rathmann NM

 ----------------------------------------------------------------------------------------------------

In this paper, we revisit the nonlinear-viscous firn rheology introduced by Gagliardini and Meyssonnier (1997)
that allows posing multi-dimensional firn densification problems subject to arbitrary stress and temperature fields.
In this sample code in particular, we reproduce the transient collapse of a Greenlandic firn tunnel as a cross-section
model. The simulation is based on the tunnel built at the NEEM drilling site during the 2012 campaign by setting the
initial dimensions and surface temperatures to the ones measured. The results are then compared to the collapse
measurements taken during the two-year-long experiment

"""

##############################################################IMPORTS
#importing modules in this particular order to avoid version conflicts
from math import *
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interpolatescipy
from scipy.interpolate import interp1d
import os

try:
    #h5py must be imported before importing fenics
    import h5py
    import gmsh
    import meshio
    
    
    #Set shortcut for calling geometry functions
    geom = gmsh.model.geo
    
    
    print('before----FENICS',h5py.__version__)
    print('before----FENICS',gmsh.__version__)
    print('before----FENICS',meshio.__version__)

except ImportError:
    print("meshio and/or gmsh not installed. Requires the non-python libraries:\n",
          "- libglu1\n - libxcursor-dev\n - libxinerama1\n And Python libraries:\n"
          " - h5py",
          " (pip3 install --no-cache-dir --no-binary=h5py h5py)\n",
          "- gmsh \n - meshio")
    exit(1)

#now we can import fenics and dolfin
from fenics import *
from dolfin import *

#control verbosity of solve function
set_log_level(21)

#for plotting and font management
from matplotlib.ticker import AutoMinorLocator
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams.update({'font.size': 12})

##############################################################


def acc_from_watereqyr_to_snoweqs(acc,rho_snow, verbose=True):
    """Change accumulation units from ice equivalent meters per year to snow equivalent meters per second"""
    
    accum_watereqyr = acc # water equiv. accum, m/yr
    accum_iceeqyr = 1000/rho_ice * accum_watereqyr
    accum_iceeqs = accum_iceeqyr/yr2s # ice accum, m/s 
    acc_rate = rho_ice/rho_snow * accum_iceeqs
    
    if verbose:
        print(f'{acc=} m of water eq per year')
        print(f'{acc_rate*yr2s=} m of snow eq per year')
        print(f'{acc_rate=} m of snow eq per second')
    
    return acc_rate



def get_ab_Z07(rho,rho_ice,phi_snow,ab_phi_lim,nglen,K):
    
    rhoh = rho/Constant(rho_ice) # normalized density (rho hat)
    rhohsnow, rhohcrit = Constant(phi_snow), Constant(ab_phi_lim)

    f_a0 = lambda rhoh: (1+2/3*(1-rhoh))*rhoh**(-2*nglen/(nglen+1))
    f_b0 = lambda rhoh: 3/4*((1/nglen*(1-rhoh)**(1/nglen))/(1-(1-rhoh)**(1/nglen)))**(2*nglen/(nglen+1))

    gamma_mu = 20*1
    mu = lambda rhoh: 1/(1+exp(-gamma_mu*(rhohcrit*1-rhoh))) # step function (approximated by logistics function)

    gamma_a = lambda k: (ln(k)-ln(f_a0(rhohcrit)))/(rhohcrit-rhohsnow)
    gamma_b = lambda k: (ln(k)-ln(f_b0(rhohcrit)))/(rhohcrit-rhohsnow)
    f_a1 = lambda rhoh,k: k*exp(-gamma_a(k)*(rhoh-rhohsnow))
    f_b1 = lambda rhoh,k: k*exp(-gamma_b(k)*(rhoh-rhohsnow))

    f_a = lambda rhoh,k: f_a0(rhoh) + mu(rhoh)*f_a1(rhoh,k)
    f_b = lambda rhoh,k: f_b0(rhoh) + mu(rhoh)*f_b1(rhoh,k)
    
    a = f_a(rhoh,K)
    b = f_b(rhoh,K)

    
    return a, b


def get_sigma(v,a,b,Aglen,nglen):
    
    eps_dot=sym(grad(v))                      
    J1=tr(eps_dot) 
    J2=inner(eps_dot,eps_dot)
    eps_E2=1/a*(J2-J1**2/3) + (3/2)*(1/b)*J1**2
    viscosity = (1/2)**((1-nglen)/(2*nglen)) * Aglen**(-1/nglen) * (eps_E2)**((1-nglen)/(2*nglen))    
    sigma = viscosity * (1/a*(eps_dot-(J1/3)*Identity(2))+(3/2)*(1/b)*J1*Identity(2))
    
    return sigma

"""
Define boundaries of the domain
In order to define the inner irregular boundary, we define it first as the whole
domain of boundaries, and start subtracting the rest of the well defined boundaries 
by properly defining them
"""

class obstacle_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class bottom_boundary(SubDomain):
    def inside(self, x, on_boundary):
        # return on_boundary and near(x[1],0)
        return on_boundary and (x[1]<1)  

class top_boundary(SubDomain):
    def inside(self,x,on_boundary):
        dz=0.1 #tolerance. how many meters from the surface's minimum height (which is updated every timestep)
        return on_boundary and (x[1]> (zmin-dz))
        #the left and right boundary nodes within this zone are not a problem because we
        # will define the other boundaries on top of these definitions
    
class right_boundary(SubDomain):
    def inside(self,x,on_boundary):
        # return on_boundary and near(x[0],L)
        return on_boundary and (x[0]>(L-0.5))

class left_boundary(SubDomain):
    def inside(self,x,on_boundary):
        # return on_boundary and near(x[0],0)  
        return on_boundary and (x[0]<0.5)
    
    
#Left and right periodic boundaries
#WATCH OUT!!!
#we need to build common left and right boundary node heights
# for the periodic boundary conditions to work. Otherwise it will ignore them
#https://oldqa.fenicsproject.org/5812/periodic-boundary-condition-for-meshes-that-are-not-built-in/

class PeriodicBoundary(SubDomain):
    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return (x[0]<0.1) and on_boundary
    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - L
        y[1] = x[1]

pbc=PeriodicBoundary()



"""-----------------------------------------------------REMESHING-------------"""

def msh2xdmf(filename,outname="mesh_temp.xdmf"):
    
    """Change the format of the mesh file we have saved"""

    msh = meshio.read(filename)
    
    #----------------------------------------------------------------------------------
    #ENSURE THAT LEFT AND RIGHT BOUNDARY NODES HAVE EXACTLY THE SAME VERTICAL COORDINATES
    #                        (for the periodic boundary conditions to work on a custom mesh)

    print(f'\n\n[[x,y,z],...<-->Nnodes]\n{np.shape(msh.points)=}\n\n')
    print(f'\n\n{msh.points=}\n\n')

    Nnodes=np.shape(msh.points)[0]

    tolerance = 1e-2  # <<<<<<<<<<1e-6 might need some hand tuning
    zs_L=np.array([])
    is_L=np.array([])
    zs_R=np.array([])
    is_R=np.array([])

    for i in range(Nnodes):

        coords=msh.points[i,:]

        if (0-tolerance)<=coords[0]<=(0+tolerance):
            #print(f' LEFT {coords=}')
            is_L = np.append(is_L, i)  # index
            zs_L=np.append(zs_L,coords[1]) #only interested in z

        elif (L-tolerance)<=coords[0]<=(L+tolerance):
            #print(f' RIGHT {coords=}')
            is_R = np.append(is_R, i)  # index
            zs_R = np.append(zs_R, coords[1])  # only interested in z

    argsort_L=np.argsort(zs_L)
    argsort_R = np.argsort(zs_R)


    print(f'{len(zs_L)=}---------{len(zs_R)=}')
    if len(zs_L)!=len(zs_R):
        print('WATCH OUT, PERIODIC BOUNDARY CONDITIONS WILL NOT WORK UNLESS THE NODES ON EITHER SIDE HAVE EXACTLY THE SAME POSITIONS')
        print('We can equalize the positions here but we cannot add or remove any nodes')
        raise Exception("lateral boundary arrays of different size\nCHANGE TOLERANCE OR REMOVE FIRST AND LAST ITEMS FROM XS_SURF AND ZS_SURF IN REMESH")

    print(f'{zs_L[argsort_L]=}')
    print(f'{zs_R[argsort_R]=}')

    zsmean_LR= (zs_L[argsort_L] + zs_R[argsort_R])/2

    print(f'{zsmean_LR=}')


    for j in range(len(zs_L)): #both the same now
        #node indexes
        i_L= int(is_L[argsort_L][j])
        i_R = int(is_R[argsort_R][j])
        #set new common value
        msh.points[i_L,:]=[0,zsmean_LR[j],0] #[x,y,z]
        msh.points[i_R,:]=[L,zsmean_LR[j],0] #[x,y,z]


    #------------------------------------------------------------------------------------

    line_cells = []
    for cell in msh.cells:
        if cell.type == "triangle":
            triangle_cells = cell.data
        elif cell.type == "line":
            if len(line_cells) == 0:
                line_cells = cell.data
            else:
                line_cells = np.vstack([line_cells, cell.data])
    
    line_data = []
    for key in msh.cell_data_dict["gmsh:physical"].keys():
        if key == "line":
            if len(line_data) == 0:
                line_data = msh.cell_data_dict["gmsh:physical"][key]
            else:
                line_data = np.vstack([line_data, msh.cell_data_dict["gmsh:physical"][key]])
        elif key == "triangle":
            triangle_data = msh.cell_data_dict["gmsh:physical"][key]
    
    triangle_mesh = meshio.Mesh(points=msh.points[:, :2], cells={"triangle": triangle_cells},
                                cell_data={"name_to_read": [triangle_data]})
    
    line_mesh = meshio.Mesh(points=msh.points[:, :2], cells=[("line", line_cells)],
                            cell_data={"name_to_read": [line_data]})
    meshio.write(outname, triangle_mesh)


def angle(x,z,xc,zc): 
    
    """Compute angle of x,z coordinates with respect to the xc,zc center coordinates"""

    if x>=xc:
        alpha= np.arctan((z-zc)/(x-xc))
    else:
        alpha= np.pi + np.arctan((z-zc)/(x-xc))
        
    if alpha>2*np.pi:
        alpha-=2*np.pi
        
    elif alpha<0:
        alpha+=2*np.pi
        
    return alpha


def dilation(xs,zs,factor=0.95):
    
    """Dilate set of points taking the mean coordinates as center of reference
    When the factor is bigger than one the image is bigger than the original
    otherwise, it is smaller
    We are interested in the latter because we use this function to order the tunnel's inner boundary nodes effectively
    """

    Npoints=len(xs)
    
    xdil=np.zeros(Npoints)
    zdil=np.zeros(Npoints)
    
    xc=np.mean(xs)
    zc=np.mean(zs)
    
    for i in range(Npoints):
        
        r=np.sqrt((xc-xs[i])**2 + (zc-zs[i])**2)
        alpha = angle(xs[i],zs[i],xc,zc)
        
        r_new= factor * r
        
        xdil[i]= xc + r_new*np.cos(alpha)
        zdil[i]= zc + r_new*np.sin(alpha)
        
    return xdil,zdil


    
def sort_hole_fenics_v2(hole_coords,ndmin=4):
        
        """
        Function that defines an order for the tunnel's boundary nodes, which is not trivial to sort since
        it is a closed loop and we can have a lot of identical xs and zs (reason why we use a polar approach)
        and the edges sometimes form bottlenecks, which makes implementing this order from the distance to the
        previous node impossible.
        We do not care where the points start, we just need them to follow a direction around
        the hole and back to the start
        
        hole_coords = U.tabulate_dof_coordinates()
        ndmin=how many nearest neighbors will be checked

        """
        #initial sort
        xs_hole_xsorted, zs_hole_xsorted, s_numdofs = sort_fenics(hole_coords,axis=0) #just for the algorithm to be quicker
        s_numdofs=len(xs_hole_xsorted)

        #dilate
        xdil,zdil =  dilation(xs_hole_xsorted, zs_hole_xsorted,factor=0.94)
        
        #points sorted to follow the loop
        xs_hole=[]
        zs_hole=[]
        
        #arbitrary initial point. (away from corner spikes)
        
        isafe=int(s_numdofs/2)
        
        xs_hole.append(xs_hole_xsorted[isafe])
        zs_hole.append(zs_hole_xsorted[isafe])
        
        x_rn = xs_hole_xsorted[isafe] #coordinates of the point we are calculating the distance from rn
        z_rn = zs_hole_xsorted[isafe]
        
        xd_rn = xdil[isafe]
        zd_rn = zdil[isafe]
        
        #delete it from the list of points to be sorted
        xs_hole_xsorted= np.delete(xs_hole_xsorted,isafe)
        zs_hole_xsorted= np.delete(zs_hole_xsorted,isafe)
        
        xdil= np.delete(xdil,isafe)
        zdil= np.delete(zdil,isafe)
        
        #calculate closest point and follow the loop from there
        #the direction will be random but not important
        #we can maybe improve the cost of this function by searching just in the closest half
        
        for ii in range(s_numdofs-1):
            
            ndmin=min(ndmin,len(xs_hole_xsorted))
            
            dist= (xs_hole_xsorted - x_rn)**2 + (zs_hole_xsorted - z_rn)**2
            i_mins=np.argsort(dist)[:ndmin] #indexes of the ndmin minimum distances
            

            ref_angle=angle(x_rn,z_rn,xd_rn,zd_rn)
            
            #angle with respect to inner close point. OXY frame of reference
            alpha_mins=np.zeros(ndmin)
            #angle with respect to inner close point. POINT_rn frame of reference
            alpha_mins_ref=np.zeros(ndmin)
            
            for i in range(ndmin):
                
                alpha_mins[i]=angle(xs_hole_xsorted[i_mins[i]],zs_hole_xsorted[i_mins[i]],xd_rn,zd_rn)
                
                dif = alpha_mins[i] - ref_angle
                if dif > 0:
                    alpha_mins_ref[i] = dif
                else:
                    alpha_mins_ref[i] = dif + 2*np.pi
                 
                
            
            i_next= i_mins[np.argmin(alpha_mins_ref)]
            
            
            #append
            xs_hole.append(xs_hole_xsorted[i_next])
            zs_hole.append(zs_hole_xsorted[i_next])
            
            x_rn = xs_hole_xsorted[i_next] #coordinates of the point we are calculating the distance from rn
            z_rn = zs_hole_xsorted[i_next]
            
            xd_rn = xdil[i_next]
            zd_rn = zdil[i_next]
            
            #delete it from the list of points to be sorted
            xs_hole_xsorted= np.delete(xs_hole_xsorted,i_next)
            zs_hole_xsorted= np.delete(zs_hole_xsorted,i_next)
            
            xdil= np.delete(xdil,i_next)
            zdil= np.delete(zdil,i_next)
            
        
        return xs_hole, zs_hole
        
        

def remesh_acc(xs_hole,zs_hole,xs_surf,zs_surf,L,tstep,dt,acc_rate,n_itacc, tmsurf=0.15,tmr=0.15,nacclayers=5,maxN=100, mode='linear',outname='mesh_temp.msh'): #jarri txukunago gero

    """
    As ALE.move() evolves the mesh, the original discretization of the space stops being appropriate.
    We need to periodically discretize the deformed domain to avoid problems

    Mode can be linear or spline. Linear cuts some corners if the number of tunnel-boundary-nodes is not too high,
    but spline smooths the shape too much

    tmsurf = defines the mesh resolution on the surface (average distance between nodes, smaller is finer)
    tmr = defines the mesh resolution on the tunnel's inner surface (average distance between nodes, smaller is finer)
    These two need to be tuned by hand because if the grid is too fine, the deformation imposed by ALEmove since the last remesh
    can be too big (especially for the biggest K values) and the grid overlaps and Fenics blows up with a [-1] error
    maxN caps the number of nodes that there can be on the tunnel boundary to try to avoid this

    Every n_itacc iterations it accumulates the meters of snow that have fallen in that time.
    how many iterations to accumulate until it can be described by a new surface node layer"""
    
    #Define size of mesh for different places
    tm = 3.0#Exterior resolution (far from tunnel). Much more regular evolution, so can be described with a coarser grid
    
    acc_iter=True if tstep%n_itacc==0 else False #flag to know if in this iteration the acumulation layer will be added
    
    deltah=n_itacc*dt*acc_rate if acc_iter else 0 #meters that should be added to account for accumulation

    
    #Initialize mesh
    gmsh.initialize()

    ######### Create tunnel curve using spline ###########
    
    N_hole=len(xs_hole)
    
    """The resolution of each point does not take into account how many 
    points it already has around, so there is a positive feedback loop
    that makes the resolution go crazy from a certain resolution to
    point density ratio. 
    need to limit the maximum number of points by hand"""
    
    # maxN=150 #to be adjusted. assemble() glitches at around 2000
    portion= (N_hole//maxN) + 1 #portion of the points to keep

    xs_hole = xs_hole[::portion]
    zs_hole = zs_hole[::portion]
    N_hole=len(xs_hole)
    
    ps_hole=[]
    
    for i in range(N_hole):
        
        ps_hole.append(geom.addPoint(xs_hole[i],zs_hole[i], 0,tmr))
    
    ps_hole.append(1) #First and last points (tags) must be the same to have a close boundary!
    

    #Create 'line' by interpolating around the give points
    
    if mode=='spline':
        
        curve_hole = geom.addBSpline(ps_hole,-1)
        #Create the curve loop of the hole
        hole = geom.addCurveLoop([curve_hole])
        
    elif mode=='linear':
        
        curve_hole=[]
        for i in range(N_hole):
            curve_hole.append(geom.addLine(ps_hole[i],ps_hole[i+1]));

        #Create the curve loop of the hole
        hole = geom.addCurveLoop(curve_hole);
    
    
    ######### Create exterior boundary using spline ###########

    H_left=zs_surf[np.argmin(xs_surf)]
    H_right=zs_surf[np.argmax(xs_surf)]
    # they must be the same down to the computation precission
    # take the mean
    H_LR = (H_left + H_right) / 2
    print(f'\n{H_left=}\n{H_right=}\n{H_LR=}\n')

    # WATCH OUT!!!
    # we need to build common left and right boundary node heights for the periodic boundary conditions to work
    Nys_LR = 50
    ys_lr = np.linspace(0, H_LR, num=Nys_LR)

    #Irregular surface (left to right)
    xs1=np.concatenate(([0],xs_surf,[L]))
    ys1 = np.concatenate(([H_LR + deltah], zs_surf + deltah, [H_LR + deltah]))  # accumulation included there
    
    #Add all the surface points to mesh
    ps_surf=[]

    for i in range(len(xs1)):
        ps_surf.append(geom.addPoint(xs1[i],ys1[i], 0,tmsurf))

    p1=geom.addPoint(L,0,0,tm)
    p2=geom.addPoint(0,0,0,tm)
    
    l1 = geom.addBSpline(ps_surf, -1)  # tag=-1 to set the tags automatically
    l2=geom.addLine(N_hole+len(xs1),p1)
    l3=geom.addLine(p1,p2)
    l4 = geom.addLine(p2, N_hole + 1)  # last automatically set tag
    
    ext=geom.addCurveLoop([l1,l2,l3,l4])

    ############ Generate the mesh itself ##############
    
    #Create surface between exterior and hole
    s = geom.addPlaneSurface([ext,hole])

    gmsh.model.addPhysicalGroup(2, [s], tag=tstep)
    gmsh.model.setPhysicalName(2, tstep, "Firn/ice")
    
    #Generate mesh and save
    
    geom.synchronize()
    gmsh.model.mesh.generate(2)
    
    gmsh.option.setNumber('Mesh.SurfaceFaces', 1)
    gmsh.option.setNumber('Mesh.Points', 1)
    
    
    gmsh.write(outname) #write msh file, but we ned xdmf to work with fenics
    msh2xdmf(outname,outname="initial_condition_data/mesh_temp_100.xdmf") #rewrite the file into an xdmf to be read
    
    gmsh.finalize() #important to close gmsh!!
    
    return acc_iter,deltah,n_itacc,xs1,ys1

def sort_fenics(scoords,axis):
    
        """Sort the unsorted arrays that Fenics works with
        
        scoords = U.tabulate_dof_coordinates()
        axis---according to which axis must be sorted---(xs=0,zs=1)
        
        """
        step=1
        
        axis_dof= scoords[:,axis] #axis coords of func space NODES (xs=0,zs=1)
        IS = np.argsort(axis_dof)
        s_numdofs= len(axis_dof)
        
        scoords_x=scoords[:,0][IS] #sorted
        scoords_z=scoords[:,1][IS]
        
        #depending on space order, points might appear doubled
        if (np.abs(s_numdofs-2*len(np.unique(axis_dof)))<2):
            
            print('doubled')
            step=2
        
        return scoords_x[::step],scoords_z[::step], int(s_numdofs/2)

    
def A_glen_Arrhenius(T):
    
    """Compute the flow factor A(T) for that temperature. This function allows for temperatures higher than -10 too"""

    R = Constant(8.314)

    #--------------------the activation energy for the creep, Q
    Qminus=60e3
    Qplus=139e3 #this is the zwinger value
    Q=conditional(le(T,-10 + 273.15),Qminus,Qplus) #LessOrEqual 
    
    #-----------------A0 is also defined in two parts
    A0minus=3.985e-13
    A0plus=1.916e3 #zwinger
    A0=conditional(le(T,-10 + 273.15),A0minus,A0plus) #LessOrEqual 

    return A0*exp(-Q/(R*T))
    
##########################################

#Define parameters
n=3 #exponent
g=Constant((0.0,-9.82)) #gravity

#coefficient functions
phi_snow= 0.4
ab_phi_lim=0.81

#density
rho_surf=307.2#NEEM, Breant et al
rho_ice= 910
rho_ice_softened= rho_ice - 0.2


#trench density and velocities parameters
rho_trench= 550 #denser firn after cutting the trench out. measured
deltax_trench=0.25#0.25 #distance from the trench to linearly smooth the density
deltaz_trench=0.25#0.25
u_trench=3.5 #upper trench
l_trench=2.25
trench_lim=3.1
bump=1 #implicit in this case, not shown as a bump
#the following are just to generate a sensible initial guess to solve the momentum equation in the first iteration
factor_inner=0.9 #0.9 #factor to account for the bigger load induced increased velocities right below the tunnel
factor_outer=1.1 #1.1 #factor to account for the bigger load induced increased velocities to the sides of the tunnel
trench_zoom_factor=1


#conversion factors
day2s=24*60*60
yr2s=365.25*day2s

#accumulation
snow550=True #accumulation in 550kg/m3 instead of snow. to avoid computational issues
acc_meas=0.41 #Measured acc in m of water equivalent per year accounting for drif snow enhanced accumulation

if snow550:
    acc_rate=acc_from_watereqyr_to_snoweqs(acc_meas,rho_snow=rho_trench) #in m 550snow/s
else:
    acc_rate=acc_from_watereqyr_to_snoweqs(acc_meas,rho_surf) #in m snow/s

# print(f' \n acc in m SNOW/yr {acc_from_watereqyr_to_snoweqs(acc_meas,rho_surf)*yr2s=}')
# print(f'  acc in m TRENCHSNOW/yr {acc_from_watereqyr_to_snoweqs(acc_meas,rho_snow=rho_trench)*yr2s=}\n')
# print(f'{acc_rate=} m snow per second')
# print(f' per step: {acc_rate*dt=} m snow per step')


#temperature
Tsite=-28.8 + 273.15 #NEEM,depth-averaged firn temperature

# Thermal parameters of pure ice
c0=2127.5
c1=7.253
T0=273.16

k0=9.828
gammaT=0.0057
kf0=0.138
kf1=1.010e-3
kf2=3.233e-6


#mesh dimensions
L = 20  # Length of channel
H = 30  # height of channel
#remeshing
acc_iter=False
nacclayers=1
tmsurf=0.3
tmr=0.15
hole_maxN=200


##########################################    USER INPUT

#Ask to define the specific case we want to run:
#             average climate  -> surface is forced with the average surface temperature measured for this period
#             variable climate -> surface is forced with the variable temperature measurements at NEEM for this period
correctinput=False
optionlist=['variable','average']
print('\nChoose whether you want to force the surface temperature with the variable temperature record or with its average:')
while not correctinput:
    flagclimate=input('      >>>write variable or average (and press enter): ')
    if flagclimate in optionlist:
        correctinput=True
    else:
        print(f'{flagclimate=} is not an option. Choose between {optionlist=}')


#Ask to define the specific case we want to run:
#             coldtrench initial temperature  -> NEEM's depth-averaged firn temperature is used as initial T
#             hottrench initial temperature  -> NEEM's depth-averaged firn temperature is used as initial T but the trench is hotter
correctinput=False
optionlist=['coldtrench','hottrench']
print('\nChoose whether you want the initial temperature of the trench to be hotter or just the depth-averaged firn temperature (as the rest of the domain):')
while not correctinput:
    flagtrench=input('      >>>write hottrench or coldtrench (and press enter): ')
    if flagtrench in optionlist:
        correctinput=True
    else:
        print(f'{flagtrench=} is not an option. Choose between {optionlist=}')


if flagtrench=='hottrench':
    print('\nChoose the HOTTER trench start temperature (or just press enter if you want to choose the default -5ºC) :')
    T0hotter=(np.float(input('      Ttrench= (in ºC and include the negative sign): ') or -5)+273.15) #NEEM,
else:
    T0hotter = Tsite #trench is at the same temperature as the rest


K=int(input('\nChoose value of K? (10-2000, press enter for default=1000): ') or 1000)


#time discretization
print('\nThe bigger K is, the faster the velocities and, thus, the smaller the timestep needs to be')
print('If we take a reasonable timestep for K=100 as the unit, choose what fraction of that timestep you want:')
print('\nRecommended values-->\n[K=100:dtf=1.0 (3 days), K=500:dtf=0.8 (2.4 days), K=1000:dtf=0.5 (1.5 days), K=2000:dtf=0.3 (1 day) ')
dtfactor=float(input('                Choose an appropriate dtfactor: ' ))
dt=(0.008333*yr2s)*dtfactor
print(f'----------------------------------------------dt={np.round(dt,2)} s')
print(f'----------------------------------------------dt={np.round(dt/(60*60*24),3)} days')

nyears_ref=1.96#yrs from the technical report.
nsteps= int( (nyears_ref+0.04)*yr2s /dt + 1 ) #added a small time buffer, but stops automatically
lastyear=-1 #just a flag


#remesh
print(f'\n Choose how often we want to remesh:')
remeshstep=int(input(f'                       introduce an appropiate remeshstep (press enter for default=4): ') or 4)
#how often we include the accumulation layer
n_itacc=remeshstep #n_itacc must be a multiple of how often we remesh


#plotting
plotting=True
nplot=int(30/dtfactor) #every how many iterations it plots

#-steady state definition
dHdt=100      #So that it enters the while
dHdt_tol=0.01 #Change to consider that we have reached the steady state

print('\n\n>>>>>>>>>>>>>>>>>>>>>>>> NO MORE USER INPUT IS REQUIRED FROM HERE ON <<<<<<<<<<<<<<<<<<<<')
print('                           Code running...\n\n')

#-------------------------------------------------------READING SURFACE TEMPERATURE RECORD-------------------

# LOAD temperatures for this period from NEEM's weather station
# period----from 2012.5273785078yr to 2014.5273785078yr (tunnel collapse data)
neem_Ts = np.load('initial_condition_data/NEEM_temperaturesTA1_smooth.npy')
neem_ts = np.load('initial_condition_data/NEEM_timesTA1.npy')
neem_t0 = neem_ts[0]

print(f'\n>>>>>>>available temperature data from:')
print(f'{np.min(neem_ts)}yr to {np.max(neem_ts)}yr \n      {np.max(neem_ts) - np.min(neem_ts)}yr')



# ---------------COMPUTE AVERAGE TEMPERATURE FOR THE STEP'S dt WINDOW
avg_Tstep = np.zeros(nsteps)
steps_ts = np.zeros(nsteps)

for ttstep in range(nsteps):

    steps_ts[ttstep] = neem_t0 + ttstep * dt / yr2s

    if ttstep > 0:

        tinit_yr = neem_t0 + (ttstep - 1) * dt / yr2s
        tfin_yr = neem_t0 + ttstep * dt / yr2s

        windowmask = (tinit_yr <= neem_ts) & (neem_ts <= tfin_yr)

        Tstep = np.mean(neem_Ts[windowmask])
        avg_Tstep[ttstep] = Tstep

    elif ttstep == 0:
        Tstep = neem_Ts[0]
        avg_Tstep[0] = Tstep

#--compute mean T
Tsurfavg=np.mean(neem_Ts)+273.15
print(f'\n\n--------average temperature for the period {Tsurfavg-273.15=}')



#-------------------------------------------MESH-----------------------------------------------------#
meshfilename="initial_condition_data/NEEM_setup_withsurfacebump.xdmf"  #with the '.xdmf'  included

mesh = Mesh()
with XDMFFile(meshfilename) as infile:  #READ FILE
    infile.read(mesh)


#----------INITIALIZE ARRAYS AND VARIABLES
tstep=0
zmin=29

igifs=0

neemsaved=False
still_not_shown_ref=True

volumes=np.array([])
hole_deltaxs=np.array([])
hole_deltazs=np.array([])
hole_zmins=np.array([])
times=np.array([])


max_Ts=np.zeros(nsteps)
min_Ts=np.zeros(nsteps)
max_RHOs=np.zeros(nsteps)
min_RHOs=np.zeros(nsteps)
max_Vxs=np.zeros(nsteps)
min_Vxs=np.zeros(nsteps)
max_Vys=np.zeros(nsteps)
min_Vys=np.zeros(nsteps)


#----------------------------------------------------------------------------------------------------
#------------------------------------MAIN LOOP----------------------------------------------------------
#----------------------------------------------------------------------------------------------------

while tstep<=nsteps:

    q_degree = 3
    dx = dx(metadata={'quadrature_degree': q_degree})  

    #----------------------------------------------------------------------REMESH
    
    if (tstep>0 and tstep%(remeshstep)==0):
        
        print('::::::::::::::REMESHING:::::::::::::::::::::::::::')
        
        #Get coordinates and velocities of the points at the obstacle's surface
        mesh.init(2,1)
        dofs = []
        cell_to_facets = mesh.topology()(2,1)
        for cell in cells(mesh):
            facets = cell_to_facets(cell.index())
            for facet in facets:
                if boundary_subdomains[facet] == 5: #We have given the number 5 to the subdomain of the obstacle
                    dofs_ = V.dofmap().entity_closure_dofs(mesh, 1, [facet])
                    for dof in dofs_:
                        dofs.append(dof)
        
        unique_dofs = np.array(list(set(dofs)), dtype=np.int32)
        hole_coords = V.tabulate_dof_coordinates()[unique_dofs] #tunnel surface node coordinates
        
    
        #Get coordinates and velocities of the points at th surface
        mesh.init(2,1)
        dofs = []
        cell_to_facets = mesh.topology()(2,1)
        for cell in cells(mesh):
            facets = cell_to_facets(cell.index())
            for facet in facets:
                if boundary_subdomains[facet] == 2: #We have given the number 2 to the subdomain of the surface
                    dofs_ = V.dofmap().entity_closure_dofs(mesh, 1, [facet])
                    for dof in dofs_:
                        dofs.append(dof)
        
        unique_dofs = np.array(list(set(dofs)), dtype=np.int32)
        surface_coords = V.tabulate_dof_coordinates()[unique_dofs] #surface node coordinates
    

        #sort the points before passing them to gmsh and the other remeshing functions
        surface_xs, surface_zs,_ = sort_fenics(surface_coords,axis=0)
        hole_xs, hole_zs = sort_hole_fenics_v2(hole_coords,ndmin=4)
        
        
        #save hole max deltax and max deltaz throughout simulation
        #Assuming that the shape of the tunnel stays relatively regular (as it does in our case)
        hole_deltaxs=np.append(hole_deltaxs, np.max(hole_xs) - np.min(hole_xs))
        hole_deltazs=np.append(hole_deltazs, np.max(hole_zs) - np.min(hole_zs))
        hole_zmins= np.append(hole_zmins,np.min(hole_zs))
        times=np.append(times,tstep*dt/yr2s)

        np.save(f'results_k{K}/{flagclimate}_{flagtrench}/times_{flagclimate}_{flagtrench}_k{K}.npy',times)
        np.save(f'results_k{K}/{flagclimate}_{flagtrench}/hole_zmins_{flagclimate}_{flagtrench}_k{K}.npy',hole_zmins)
        np.save(f'results_k{K}/{flagclimate}_{flagtrench}/hole_deltaxs_{flagclimate}_{flagtrench}_k{K}.npy',hole_deltaxs)
        np.save(f'results_k{K}/{flagclimate}_{flagtrench}/hole_deltazs_{flagclimate}_{flagtrench}_k{K}.npy',hole_deltazs)
        
        
        print('----------------number of nodes forming the tunnel boundary---------',len(hole_xs))
        
        #---------------------------- REMESH<<<<<<<<<<<<<
        
        #create new mesh file
        temp_meshfile='initial_condition_data/mesh_temp_100'
        acc_iter,deltah,n_itacc,x_snowed_surf,z_snowed_surf = remesh_acc(hole_xs,hole_zs,surface_xs,surface_zs,L,
                                     tstep,dt=dt,acc_rate=acc_rate,n_itacc=n_itacc, tmsurf=tmsurf,tmr=tmr,nacclayers=nacclayers,maxN=hole_maxN,mode='linear',
                                     outname=temp_meshfile+'.msh')
        
        #read new mesh file
        mesh = Mesh()
        with XDMFFile(temp_meshfile+'.xdmf') as infile:
            infile.read(mesh)


    #--------------------------------------BOUNDARY SUBDOMAINS-------------------------------------------#
    
    #Give a number to each different boundary
    boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_subdomains.set_all(0)

    
    obstacle=obstacle_boundary()
    obstacle.mark(boundary_subdomains, 5)
    bottom=bottom_boundary()
    bottom.mark(boundary_subdomains, 1)
    top=top_boundary()
    top.mark(boundary_subdomains, 2)
    left=left_boundary()
    left.mark(boundary_subdomains, 3)
    right=right_boundary()
    right.mark(boundary_subdomains, 4)
    
    #--------------------------------------FUNCTION SPACE------------------------------------------------#
    
    #Define function space for density
    deg=2 #Polinomial degree
    U=FunctionSpace(mesh, 'Lagrange', deg, constrained_domain=pbc) # Polynomial function space of order "deg"
    rho=Function(U) # the unknown function
    wr=TestFunction(U)  # the weight function

    #Define function space for velocity
    deg=2
    V=VectorFunctionSpace(mesh, "Lagrange", deg, constrained_domain=pbc)
    v=Function(V) # the unknown function
    wv=TestFunction(V)  # the weight function 
    
    #Define function space for temperature
    deg=2#Polinomial degree
    Q=FunctionSpace(mesh, 'Lagrange', deg, constrained_domain=pbc) # Polynomial function space of order "deg"
    T=Function(Q) # the unknown function
    wT=TestFunction(Q)  # the weight function
    
    #----------------------------------------------------------------BOUNDARY CONDITIONS    
        
    #-----------------------------------TOP
    #bc_rho_s=DirichletBC(U,rho_surf,boundary_subdomains,2) #Density at the surface
    bc_rho_s=DirichletBC(U,rho_trench,boundary_subdomains,2) #denseSNOWDensity at the surface

    if flagclimate=='average':
        bc_T_t=DirichletBC(Q,Tsurfavg,boundary_subdomains,2) #average surface temperature #T at the top
    elif flagclimate=='variable':
        bc_T_t=DirichletBC(Q,avg_Tstep[tstep] + 273.15,boundary_subdomains,2) #variable surface temperature #T at the top
    #-----------------------------------BOTTOM
    bc_v_b=DirichletBC(V,(0.0,0.0),boundary_subdomains,1) #Velocity at the bottom
    bc_T_b=DirichletBC(Q,Tsite,boundary_subdomains,1) #T at the bottom
    #-----------------------------------LEFT
    bc_v_l=DirichletBC(V.sub(0),0.0,boundary_subdomains,3) #Velocity at the left boundary
    bc_T_l=DirichletBC(Q,Tsite,boundary_subdomains,3) #T at left
    #-----------------------------------RIGHT
    bc_v_r=DirichletBC(V.sub(0),0.0,boundary_subdomains,4) #Velocity at the right boundary
    bc_T_r=DirichletBC(Q,Tsite,boundary_subdomains,4) #T at right

    bcs_rho=[bc_rho_s]
    bcs_v=[bc_v_b]
    bcs_T=[bc_T_b,bc_T_t]
        
    
    #--------------------------------------INITIAL CONDITION--------------------------------------------#
    
    if tstep==0:
        
        #Get coordinates and velocities of the points at the obstacle's surface
        mesh.init(2,1)
        dofs = []
        cell_to_facets = mesh.topology()(2,1)
        for cell in cells(mesh):
            facets = cell_to_facets(cell.index())
            for facet in facets:
                if boundary_subdomains[facet] == 5: #We have given the number 5 to the subdomain of the obstacle
                    dofs_ = V.dofmap().entity_closure_dofs(mesh, 1, [facet])
                    for dof in dofs_:
                        dofs.append(dof)
        
        unique_dofs = np.array(list(set(dofs)), dtype=np.int32)
        boundary_coords = V.tabulate_dof_coordinates()[unique_dofs] #surface node coordinates
        hole_zmin=np.min(boundary_coords[:,1])
        print('--:::HOLE-zmin=',hole_zmin)
        hole_xmin=np.min(boundary_coords[:,0])
        hole_xmax=np.max(boundary_coords[:,0])
        print('--:::HOLE-xlims=(',hole_xmin,',',hole_xmax,')')
        
        iorder= np.argsort(boundary_coords[:,0])
        initial_hole_xs = boundary_coords[:,0][iorder] #for plotting it afterwards
        initial_hole_zs = boundary_coords[:,1][iorder] #for plotting it afterwards
        


        #----------INITIAL CONDITIONS------to properly order the nodes before setting the values--------------------

        #we need to provide reasonable first guesses in order to solve the equations
        #it does not impact the results, it just enables the first iteration's convergence

        #-----------------------------------initial density profile--solution of 1d problem---
        #compacted already

        #The initial background density field will be the original density profile measured at NEEM (but smoothed)
        rho_init_neem=np.load('initial_condition_data/neem_densities_raw_smooth.npy')
        z_init_neem=np.load('initial_condition_data/neem_densities_raw_smooth_zcoors.npy')
        #The initial velocity proposal is the 1D SS solution for k=200 (lowest RMSE for NEEM)
        v_init_neem=np.load('initial_condition_data/neem_K=200_SSvelocities.npy')
        zv_init_neem=np.load('initial_condition_data/neem_K=200_SSvelocities_zcoors.npy')

        #The initial trench velocity proposal is the 1D SS solution for k=200 for a 550 unburdened slab
        #but the SS solution contains more than just the surface, so we need to filter the surface area applicable to a
        #real pure 550kg volume
        #(k=200 was chosen because it's the one that minimizes NEEM's 1D RMSE)
        v_init_trench=np.load('initial_condition_data/vK200_trench550_NEEM.npy')
        zv_init_trench=np.load('initial_condition_data/ZvK200_trench550_NEEM.npy')
        #to properly understand how to scale this surface velocities we need the densities it comes from.
        #we will ZOOM into the surface and interpolate in an appropiate area to form the initial V guess

        #rho_init_trench=np.load('initial_condition_data/rhoK200_trench550_NEEM.npy')
        #zrho_init_trench=np.load('initial_condition_data/ZrhoK200_trench550_NEEM.npy')

        #----------------------------------------------------------------------
        #adjusting to mesh coordinate system (bump surface starting at 30, unperturbed surface at 29)
        z_init_neem_adjusted = z_init_neem - (np.max(z_init_neem) - H + bump)  # adjusted to the new mesh height
        zv_init_neem_adjusted = zv_init_neem - (np.max(zv_init_neem) - H + bump)  # adjusted to the new mesh height
        zv_init_trench_adjusted = zv_init_trench - (np.max(zv_init_trench) - H + bump)  # adjusted to the new mesh height
        #zrho_init_trench_adjusted = zrho_init_trench - (np.max(zrho_init_trench) - H + bump)  # adjusted to the new mesh height

        #subtract the Bottom Boundary Condition -acc(rho_snow/rho_ice) velocity
        #because, unlike the 1D SS moving window, we have a fixed bottom now
        #we have just checked that Zs go from bottom to surface so,
        v_init_neem_nobc= v_init_neem - v_init_neem[0]
        v_init_trench_nobc= v_init_trench - v_init_trench[0]

        #we will interpolate to extract the densities and velocities at the depths of interest
        f_rho_neem = interp1d(z_init_neem_adjusted, rho_init_neem, kind='cubic', fill_value='extrapolate')
        f_v_neem = interp1d(zv_init_neem_adjusted, v_init_neem_nobc, kind='cubic', fill_value='extrapolate')
        f_v_trench = interp1d(zv_init_trench_adjusted, v_init_trench_nobc, kind='cubic', fill_value='extrapolate')
        #they all start from 29m, but we will use the last one by zooming on the top 2ms

        # ------------------extracting the coordinates out of the function spaces---
        # ----------------------------------------------------------------------------

        #-------------------------------------RHO
        scoords_r0 = U.tabulate_dof_coordinates()
        xs_dof_r0 = scoords_r0[:,0] # x coords of func space NODES
        zs_dof_r0 = scoords_r0[:,1] # z coords of func space NODES
        s_numdofs_r0 = len(zs_dof_r0)
        ISz_r0 = np.argsort(zs_dof_r0)  

        #-------------------------------------V
        scoords_v = V.tabulate_dof_coordinates()
        xs_dof_v = scoords_v[:,0] # x coords of func space NODES
        zs_dof_v = scoords_v[:,1] # z coords of func space NODES
        s_numdofs_v = len(zs_dof_v)
        ISz_v = np.argsort(zs_dof_v)
        
        #-------------------------------------T
        scoords_T0 = Q.tabulate_dof_coordinates()
        xs_dof_T0 = scoords_T0[:,0] # x coords of func space NODES
        zs_dof_T0 = scoords_T0[:,1] # z coords of func space NODES
        s_numdofs_T0 = len(zs_dof_T0)
        ISz_T0 = np.argsort(zs_dof_T0)  

        #-----------------------------------------------------------------

        #intialize r_init and v_init
        r_init=Function(U)
        v_init=Function(V)
        T_init=Function(Q)
        
        # """
        # r_init.vector()=[r_init0, r_init1, r_init2....r_init267333]
        # v_init.vector()=[vx_init0, vy_init0,vx_init1,vy_init1,...vx_init267333,vy_init267333]
        #              baina fijatu abiadurak dauzkala 2*267333 elementu
        #              """

        #set the special initial condition values around the trench and tunnel
        for ii in range(s_numdofs_r0):
            
            #coordinates (common for both arrays)        
            xii = xs_dof_r0[ii]# x coord of DOF (node)
            zii = zs_dof_r0[ii] # z coord of DOF (node)
            
            #index for v_init.vector() (two elements for each one in rho.vector(), vx and vy)
            jjx=int(2*ii)
            jjy=int(2*ii + 1)
            v_init.vector()[jjx]=0.0 #because no horizontal velocities in the initial guess
            
            #------------- To linearly smooth the initial density and temperature fields
            T_init.vector()[ii]= Tsite
            deltaT_trench = T0hotter - Tsite  # to smooth out the transition
            deltarho_trench= rho_trench - f_rho_neem(zii) #to smooth out the transition


            #----------------------Set values depending on where the node falls.
            # Cases:
            if ((xii>= (L/2-l_trench) ) and (xii<= (L/2+l_trench) ) and (zii>=hole_zmin)):
                
                T_init.vector()[ii]=T0hotter
                r_init.vector()[ii]= rho_trench

                zref= H-zii #how many meters from surface
                zbase=hole_zmin #trench start depth

                #approximated first guess
                v_init.vector()[jjy]= factor_inner*(f_v_neem(zbase)-f_v_neem(0)) \
                                      + f_v_trench(H-bump-zref/trench_zoom_factor)\
                                      - f_v_trench(H-bump-zbase/trench_zoom_factor) 
                                      #because we want the relative (what we must add)

            elif ((xii>= (L/2-u_trench) ) and (xii<= (L/2+u_trench) ) and (zii>=zmin-trench_lim)):
                
                T_init.vector()[ii]=T0hotter
                r_init.vector()[ii]= rho_trench
                
                zref= H-zii
                zbase=zmin-trench_lim

                v_init.vector()[jjy]= factor_outer*(f_v_neem(zbase)-f_v_neem(0)) \
                                      + f_v_trench(H-bump-zref/trench_zoom_factor)\
                                      - f_v_trench(H-bump-zbase/trench_zoom_factor)

            elif ((xii>= (L/2-u_trench-bump) ) and (xii<= (L/2+u_trench+bump) ) and (zii>=zmin)):
                
                T_init.vector()[ii]=T0hotter
                r_init.vector()[ii]= rho_trench
                
                zref= H-zii
                zbase=zmin

                v_init.vector()[jjy]= 1*(f_v_neem(zbase)-f_v_neem(0)) \
                                      + f_v_trench(H-bump-zref/trench_zoom_factor)\
                                      - f_v_trench(H-bump-zbase/trench_zoom_factor) 

           #---------------upper trench simple smoothing
            elif ((xii>= (L/2-u_trench - deltax_trench) ) and (xii<= (L/2-u_trench) ) and (zii>=zmin-trench_lim) and (zii<=zmin-deltaz_trench)):

                T_init.vector()[ii]=Tsite + ((xii-(L/2-u_trench - deltax_trench))/deltax_trench) * deltaT_trench
                r_init.vector()[ii]= f_rho_neem(zii) + ((xii-(L/2-u_trench - deltax_trench))/deltax_trench) *deltarho_trench  #linear transition for now
                #neglecting the velocity's smoothing
                v_init.vector()[jjy]= 1*(f_v_neem(zii)-f_v_neem(0)) 
                
            elif ((xii>= (L/2+u_trench) ) and (xii<= (L/2+u_trench + deltax_trench) ) and (zii>=zmin-trench_lim) and (zii<=zmin-deltaz_trench)):

                T_init.vector()[ii] = Tsite - ((xii-(L/2+u_trench + deltax_trench))/deltax_trench) * deltaT_trench
                r_init.vector()[ii]= f_rho_neem(zii) - ((xii-(L/2+u_trench + deltax_trench))/deltax_trench) *deltarho_trench
                v_init.vector()[jjy]= 1*(f_v_neem(zii)-f_v_neem(0)) 
                
           #---------------------- lower trench smoothing ----------------------------------------------------------------------------
            elif ((xii>= (L/2-l_trench - deltax_trench) ) and (xii<= (L/2-l_trench) ) and (zii>=hole_zmin) and (zii<=zmin-trench_lim-deltaz_trench)):

                T_init.vector()[ii] = Tsite + ((xii-(L/2-l_trench - deltax_trench))/deltax_trench) * deltaT_trench
                r_init.vector()[ii]= f_rho_neem(zii) + ((xii-(L/2-l_trench - deltax_trench))/deltax_trench) *deltarho_trench
                v_init.vector()[jjy]= factor_outer*(f_v_neem(zii)-f_v_neem(0)) 
                
            elif ((xii>= (L/2+l_trench) ) and (xii<= (L/2+l_trench + deltax_trench) ) and (zii>=hole_zmin) and (zii<=zmin-trench_lim-deltaz_trench)):

                T_init.vector()[ii] = Tsite - ((xii-(L/2+l_trench + deltax_trench))/deltax_trench) * deltaT_trench
                r_init.vector()[ii]= f_rho_neem(zii) - ((xii-(L/2+l_trench + deltax_trench))/deltax_trench) *deltarho_trench
                v_init.vector()[jjy]= factor_outer*(f_v_neem(zii)-f_v_neem(0)) 
                
            #-------------------------------------------------------------------------------------------------------
            elif ((xii>= 0 ) and (xii<= (L/2-u_trench-deltax_trench) ) and (zii>=zmin-deltaz_trench) and (zii<=zmin)):

                T_init.vector()[ii] = Tsite + ((zii-(zmin - deltaz_trench))/deltaz_trench) * deltaT_trench
                r_init.vector()[ii]= f_rho_neem(zii) + ((zii-(zmin - deltaz_trench))/deltaz_trench) *deltarho_trench
                v_init.vector()[jjy]= 1*(f_v_neem(zii)-f_v_neem(0)) 

            elif ((xii>= (L/2+u_trench+deltax_trench) ) and (xii<= L ) and (zii>=zmin-deltaz_trench) and (zii<=zmin)):

                T_init.vector()[ii] = Tsite + ((zii-(zmin - deltaz_trench))/deltaz_trench) * deltaT_trench
                r_init.vector()[ii]= f_rho_neem(zii) + ((zii-(zmin - deltaz_trench))/deltaz_trench) *deltarho_trench
                v_init.vector()[jjy]= 1*(f_v_neem(zii)-f_v_neem(0)) 
                
            #-------------------------------------------------------------------------------------------------------
            elif ((xii>= (L/2-u_trench) ) and (xii<= (L/2-l_trench - deltax_trench) ) and (zii>=zmin-trench_lim-deltaz_trench) and (zii<=zmin-trench_lim)):

                T_init.vector()[ii] = Tsite + ((zii-(zmin-trench_lim - deltaz_trench))/deltaz_trench) * deltaT_trench
                r_init.vector()[ii]= f_rho_neem(zii) + ((zii-(zmin-trench_lim - deltaz_trench))/deltaz_trench) *deltarho_trench
                v_init.vector()[jjy]= factor_outer*(f_v_neem(zii)-f_v_neem(0)) 
                
            elif ((xii>= (L/2+l_trench+deltax_trench) ) and (xii<= (L/2+u_trench) )  and (zii>=zmin-trench_lim-deltaz_trench) and (zii<=zmin-trench_lim)):

                T_init.vector()[ii] = Tsite + ((zii-(zmin-trench_lim - deltaz_trench))/deltaz_trench) * deltaT_trench
                r_init.vector()[ii]= f_rho_neem(zii) + ((zii-(zmin-trench_lim - deltaz_trench))/deltaz_trench) *deltarho_trench
                v_init.vector()[jjy]= factor_outer*(f_v_neem(zii)-f_v_neem(0)) 
                
            #------------------------------------------------------------------------
            elif ((xii>= (L/2-l_trench) ) and (xii<= (L/2+l_trench) ) and (zii>=hole_zmin-deltaz_trench) and (zii<=hole_zmin)):

                T_init.vector()[ii] = Tsite + ((zii-(hole_zmin - deltaz_trench))/deltaz_trench) * deltaT_trench
                r_init.vector()[ii]= f_rho_neem(zii) + ((zii-(hole_zmin - deltaz_trench))/deltaz_trench) *deltarho_trench
                v_init.vector()[jjy]= factor_inner*(f_v_neem(zii)-f_v_neem(0)) 
                
            #----------------------------------------------------------------------
            elif ((xii>= (L/2-l_trench-deltax_trench) ) and (xii<= (L/2-l_trench) ) and (zii>=hole_zmin-deltaz_trench) and (zii<=hole_zmin)):

                T_init.vector()[ii] = min( Tsite + ((xii-(L/2-l_trench - deltax_trench))/deltax_trench) *deltaT_trench,
                               Tsite + ((zii-(hole_zmin - deltaz_trench))/deltaz_trench) *deltaT_trench)
                
                r_init.vector()[ii]= min( f_rho_neem(zii) + ((xii-(L/2-l_trench - deltax_trench))/deltax_trench) *deltarho_trench,
                               f_rho_neem(zii) + ((zii-(hole_zmin - deltaz_trench))/deltaz_trench) *deltarho_trench)

                v_init.vector()[jjy]= factor_outer*(f_v_neem(zii)-f_v_neem(0))

                
            elif ((xii>= (L/2+l_trench) ) and (xii<= (L/2+l_trench+deltax_trench) ) and (zii>=hole_zmin-deltaz_trench) and (zii<=hole_zmin)):

                T_init.vector()[ii] = min( Tsite - ((xii-(L/2+l_trench + deltax_trench))/deltax_trench) *deltaT_trench,
                               Tsite + ((zii-(hole_zmin - deltaz_trench))/deltaz_trench) *deltaT_trench)
                
                r_init.vector()[ii]= min( f_rho_neem(zii) - ((xii-(L/2+l_trench + deltax_trench))/deltax_trench) *deltarho_trench,
                               f_rho_neem(zii) + ((zii-(hole_zmin - deltaz_trench))/deltaz_trench) *deltarho_trench)

                v_init.vector()[jjy]= factor_outer*(f_v_neem(zii)-f_v_neem(0)) 
                
            #----------------------------------------------------------------------
            elif ((xii>= (L/2-l_trench-deltax_trench) ) and (xii<= (L/2-l_trench) ) and (zii>=zmin-trench_lim-deltaz_trench) and (zii<=zmin-trench_lim)):

                T_init.vector()[ii] = max( Tsite + ((xii-(L/2-l_trench - deltax_trench))/deltax_trench) *deltaT_trench,
                               Tsite + ((zii-(zmin-trench_lim - deltaz_trench))/deltaz_trench) *deltaT_trench)
                
                r_init.vector()[ii]= max( f_rho_neem(zii) + ((xii-(L/2-l_trench - deltax_trench))/deltax_trench) *deltarho_trench,
                               f_rho_neem(zii) + ((zii-(zmin-trench_lim - deltaz_trench))/deltaz_trench) *deltarho_trench)

                v_init.vector()[jjy]= factor_outer*(f_v_neem(zii)-f_v_neem(0))

                
            elif ((xii>= (L/2+l_trench) ) and (xii<= (L/2+l_trench+deltax_trench) ) and (zii>=zmin-trench_lim-deltaz_trench) and (zii<=zmin-trench_lim)):

                T_init.vector()[ii] = max( Tsite - ((xii-(L/2+l_trench + deltax_trench))/deltax_trench) *deltaT_trench,
                               Tsite + ((zii-(zmin-trench_lim - deltaz_trench))/deltaz_trench) *deltaT_trench)
                
                r_init.vector()[ii]= max( f_rho_neem(zii) - ((xii-(L/2+l_trench + deltax_trench))/deltax_trench) *deltarho_trench,
                               f_rho_neem(zii) + ((zii-(zmin-trench_lim - deltaz_trench))/deltaz_trench) *deltarho_trench)

                v_init.vector()[jjy]= factor_outer*(f_v_neem(zii)-f_v_neem(0)) 
                
            #----------------------------------------------------------------------
            elif ((xii>= (L/2-u_trench-deltax_trench) ) and (xii<= (L/2-u_trench) ) and (zii>=zmin-trench_lim-deltaz_trench) and (zii<=zmin-trench_lim)):

                T_init.vector()[ii] = min( Tsite + ((xii-(L/2-u_trench - deltax_trench))/deltax_trench) *deltaT_trench,
                               Tsite + ((zii-(zmin-trench_lim - deltaz_trench))/deltaz_trench) *deltaT_trench)
                
                r_init.vector()[ii]= min( f_rho_neem(zii) + ((xii-(L/2-u_trench - deltax_trench))/deltax_trench) *deltarho_trench,
                               f_rho_neem(zii) + ((zii-(zmin-trench_lim - deltaz_trench))/deltaz_trench) *deltarho_trench)

                v_init.vector()[jjy]= 1*(f_v_neem(zii)-f_v_neem(0))

                
            elif ((xii>= (L/2+u_trench) ) and (xii<= (L/2+u_trench+deltax_trench) ) and (zii>=zmin-trench_lim-deltaz_trench) and (zii<=zmin-trench_lim)):

                T_init.vector()[ii] =min( Tsite - ((xii-(L/2+u_trench + deltax_trench))/deltax_trench) *deltaT_trench,
                               Tsite + ((zii-(zmin-trench_lim - deltaz_trench))/deltaz_trench) *deltaT_trench)
                
                r_init.vector()[ii]= min( f_rho_neem(zii) - ((xii-(L/2+u_trench + deltax_trench))/deltax_trench) *deltarho_trench,
                               f_rho_neem(zii) + ((zii-(zmin-trench_lim - deltaz_trench))/deltaz_trench) *deltarho_trench)

                v_init.vector()[jjy]= 1*(f_v_neem(zii)-f_v_neem(0)) 
                
            #----------------------------------------------------------------------
            elif ((xii>= (L/2-u_trench-deltax_trench) ) and (xii<= (L/2-u_trench) ) and (zii>=zmin-deltaz_trench) and (zii<=zmin)):

                T_init.vector()[ii] = max( Tsite + ((xii-(L/2-u_trench - deltax_trench))/deltax_trench) *deltaT_trench,
                               Tsite + ((zii-(zmin - deltaz_trench))/deltaz_trench) *deltaT_trench)
                
                r_init.vector()[ii]= max( f_rho_neem(zii) + ((xii-(L/2-u_trench - deltax_trench))/deltax_trench) *deltarho_trench,
                               f_rho_neem(zii) + ((zii-(zmin - deltaz_trench))/deltaz_trench) *deltarho_trench)

                v_init.vector()[jjy]= 1*(f_v_neem(zii)-f_v_neem(0))

                
            elif ((xii>= (L/2+u_trench) ) and (xii<= (L/2+u_trench+deltax_trench) ) and (zii>=zmin-deltaz_trench) and (zii<=zmin)):

                T_init.vector()[ii] = max( Tsite - ((xii-(L/2+u_trench + deltax_trench))/deltax_trench) *deltaT_trench,
                               Tsite + ((zii-(zmin - deltaz_trench))/deltaz_trench) *deltaT_trench)
                
                r_init.vector()[ii]= max( f_rho_neem(zii) - ((xii-(L/2+u_trench + deltax_trench))/deltax_trench) *deltarho_trench,
                               f_rho_neem(zii) + ((zii-(zmin - deltaz_trench))/deltaz_trench) *deltarho_trench)

                v_init.vector()[jjy]= 1*(f_v_neem(zii)-f_v_neem(0)) 
                
            #--------------------------------------------------------------------
            #from here on all profiles are a combination of NEEM's
            else:

                r_init.vector()[ii]=f_rho_neem(zii) #NEEM's measurements
                #temperatures already defined

                #what needs to be changed by hand is the velocities around the tunnel
                #vertical dimension already filtered in the previous cases
                if ((xii>= (L/2-l_trench) ) and (xii<= (L/2+l_trench) )):
                    
                    v_init.vector()[jjy]= factor_inner*(f_v_neem(zii)-f_v_neem(0))

                elif ((xii>= (L/2-u_trench-deltax_trench) ) and (xii<= (L/2-l_trench))):
                    
                    v_init.vector()[jjy]= factor_outer*(f_v_neem(zii)-f_v_neem(0))

                else:
                    
                    v_init.vector()[jjy]= 1*(f_v_neem(zii)-f_v_neem(0))

        # #--------------------------------------
        rho.assign(r_init)
        rho_prev=Function(U)
        rho_prev.assign(r_init)
        
        v.assign(v_init)
        
        T.assign(T_init)
        T_prev=Function(Q)
        T_prev.assign(T_init)

        print('\n \n -----------------------------Finished initial state definition----------\n\n')
    
    else:        
            
        rho_prev.set_allow_extrapolation(True)       
        v_sol.set_allow_extrapolation(True)
        T_prev.set_allow_extrapolation(True)    
        
        rho_init = interpolate(rho_prev,U)
        v_init = interpolate(v_sol,V)
        T_init = interpolate(T_prev,Q)
        
        rho.assign(rho_init)
        v.assign(v_init)
        T.assign(T_init)
            
    
    #--------------------------------------INTERPOLATE RHO------------------------------------------------#
    
    if tstep >0: 
        
        #:::::::::::::::::::::::::::::::::::::update density

        rho_old=rho_prev.copy()
        rho_old.set_allow_extrapolation(True)

        rho_new=Function(U)
        LagrangeInterpolator.interpolate(rho_new,rho_old)

        rho_prev.assign(rho_new)
        rho_prev.set_allow_extrapolation(True)            
        
        #................................KEEP IT BELOW RHO_ICE...........................#
        
        rhovec = rho_prev.vector()[:]
        rhovec[rhovec > rho_ice_softened] = rho_ice_softened
        rho_prev.vector()[:] = rhovec
        
        #::::::::::::::::::::::::::::::::::::::::::update temperature too
        #----------------------------------temperature
        T_old=T_prev.copy()
        T_old.set_allow_extrapolation(True)

        T_new=Function(Q)
        LagrangeInterpolator.interpolate(T_new,T_old)

        T_prev.assign(T_new)
        T_prev.set_allow_extrapolation(True)            

        #.........................................IMPOSE SNOW
        
        if acc_iter:
               
            r_snowed=Function(U)
            r_snowed = interpolate(rho_prev,U)

            #------------------------------save rho_snow on top if it lies at more than a deltah distance from surface
            #surface is identified in remesh_acc, and that's why we need the accumulation iteration to happen
            #in a remesh iteration
            
            #surface profile (including deltah)
            #WATCH OUT, LAST ELEMENT OF THE LIST IS DUPLICATED DUE TO PERIODIC BOUNDARY CONDITIONS (removed now)
            f_snowed_surface=interp1d(x_snowed_surf[:-1],z_snowed_surf[:-1],kind='cubic',fill_value='extrapolate')


            scoords_r0 = U.tabulate_dof_coordinates()
            xs_dof_r0 = scoords_r0[:,0] # x coords of func space NODES
            zs_dof_r0 = scoords_r0[:,1] # z coords of func space NODES
            s_numdofs_r0 = len(zs_dof_r0)
            ISz_r0 = np.argsort(zs_dof_r0)  
            
            
            for ii in range(s_numdofs_r0):
                        
                xii = xs_dof_r0[ii]# x coord of DOF (node)
                zii = zs_dof_r0[ii] # z coord of DOF (node)
                
                snowed= (zii > (f_snowed_surface(xii)-deltah))
                
                if snowed:
                    if snow550:
                        r_snowed.vector()[ii]= rho_trench
                    else:
                        r_snowed.vector()[ii]= rho_surf

            rho_prev = interpolate(r_snowed,U)

    rho_prev=project(rho_prev,U)
    T_prev=project(T_prev,Q)
    
    #!!!#################################################################  MAIN

    #terminal logging
    print('\n',tstep,'/',nsteps,'--------------------------------------------t=',tstep*dt/yr2s,' years')
    print(f'                                           T_average_step={avg_Tstep[tstep]}')
    print(f'                                         {acc_iter=}')
    try:
        print(f'                                     last tunnel height= {np.round(hole_deltazs[-1],3)}m')
    except:
        print('no hole_deltazs defined yet')

    #-----------------------------------------------------------------------------------------------------#
    #--------------------------------------SOLVE FEM PROBLEM----------------------------------------------#
    #-----------------------------------------------------------------------------------------------------# 
    
    #--------------------------------GET a, b, VARIABLES AND SIGMA-----------------------------------------
    print('                                    >>> computing a,b, Aglen(T), and sigma...')

    a_,b_=get_ab_Z07(rho_prev,rho_ice,phi_snow,ab_phi_lim,n,K) #Zwinger2007
    Aglen=A_glen_Arrhenius(T_prev) #<------------- Zwinger 2007 values
    sigma=get_sigma(v,a_,b_,Aglen,n)


    #-----------------------------------SOLVE MOMENTUM EQUATION--------------------------------------------
    print('                                    >>> Solving MOMENTUM equation...')

    a_v = inner(sigma,grad(wv))*dx
    L_v = rho_prev*inner(g,wv)*dx 
    
    F_v = a_v - L_v

    tol, relax, maxiter = 1e-2, 0.35, 100
    solparams = {"newton_solver":  {"relaxation_parameter":relax, "relative_tolerance": tol, "maximum_iterations":maxiter} }
    solve(F_v==0, v, bcs_v, solver_parameters=solparams)
    
    v_sol=project(v,V)

    #saving some values for monitoring the simulation
    # """
    # r_init.vector()=[r_init0, r_init1, r_init2....r_init267333]
    # v_init.vector()=[vx_init0, vy_init0,vx_init1,vy_init1,...vx_init267333,vy_init267333]
    #              baina fijatu abiadurak dauzkala 2*267333 elementu
    #              """
    max_Vxs[tstep] = max(v.vector()[0::2])*yr2s
    min_Vxs[tstep] = min(v.vector()[0::2])*yr2s
    max_Vys[tstep] = max(v.vector()[1::2]) * yr2s
    min_Vys[tstep] = min(v.vector()[1::2]) * yr2s


    #---------------------------------SOLVE MASS BALANCE EQUATION------------------------------------------
    print('                                    >>> Solving MASS BALANCE equation...')

    alpha_diff=Constant(1e-9) #FORK=1000 factor for the diffusive term. To be adjusted. 
    
    a_rho = Constant(1/dt)*rho*wr*dx + 0.5*rho*div(v_sol)*wr*dx + 0.5*dot(v_sol,grad(rho))*wr*dx + alpha_diff*dot(grad(rho),grad(wr))*dx
    L_rho =  Constant(1/dt)*rho_prev * wr *dx - 0.5*rho_prev*div(v_sol)*wr*dx - 0.5*dot(v_sol,grad(rho_prev))*wr*dx - alpha_diff*dot(grad(rho_prev),grad(wr))*dx
    
    F_rho = a_rho - L_rho

    tol, relax, maxiter = 1e-2, 0.35, 100
    
    solparams = {"newton_solver":  {"relaxation_parameter":relax, "relative_tolerance": tol, "maximum_iterations":maxiter} }
    solve(F_rho==0, rho, bcs_rho, solver_parameters=solparams)
    
    rho_prev.assign(rho)  #update rho prfile

    #saving some values for monitoring the simulation
    max_RHOs[tstep] = max(rho.vector())
    min_RHOs[tstep] = min(rho.vector())

    
    #-----------------------------------SOLVE HEAT EQUATION--------------------------------------------
    print('                                    >>> Solving HEAT equation...')

    a_,b_=get_ab_Z07(rho_prev,rho_ice,phi_snow,ab_phi_lim,n,K)
    sigma=get_sigma(v_sol,a_,b_,Aglen,n)
    eps=sym(grad(v_sol))
    ssol=inner(sigma,eps)

    cT=c0+c1*(T-T0)
    kT=(kf0-kf1*rho_prev+kf2*rho_prev**2)/(kf0-kf1*rho_ice+kf2*rho_ice**2) * k0*exp(-gammaT*T)

    a_T = T*wT*dx + dt*kT/(rho*cT)*inner(grad(T),grad(wT))*dx + dt*dot(v_sol,grad(T))*wT*dx
    L_T = T_prev*wT*dx + dt/(rho*cT)*ssol*wT*dx
    
    F_T = a_T - L_T
    
    tol, relax, maxiter = 1e-2, 0.35, 100
    solparams = {"newton_solver":  {"relaxation_parameter":relax, "relative_tolerance": tol, "maximum_iterations":maxiter} }
    solve(F_T==0,T, bcs_T, solver_parameters=solparams)
    
    T_prev.assign(T)  #update T profile
    
    #saving some values for monitoring the simulation
    max_Ts[tstep]=max(T.vector())-273.15 #Ctan
    min_Ts[tstep]=min(T.vector())-273.15


    #-----------------------------------------------------------------------------------------------------#
    #--------------------------------------------EVOLVE MESH----------------------------------------------#
    #-----------------------------------------------------------------------------------------------------#
    print('                                    >>> EVOLVING mesh...')
    
    #Get coordinates and velocities of the points at the obstacle's surface assemble
    mesh.init(2,1)
    dofs = []
    cell_to_facets = mesh.topology()(2,1)
    for cell in cells(mesh):
        facets = cell_to_facets(cell.index())
        for facet in facets:
            if boundary_subdomains[facet] == 2: #We have given the number 5 to the subdomain of the obstacle
                dofs_ = V.dofmap().entity_closure_dofs(mesh, 1, [facet])
                for dof in dofs_:
                    dofs.append(dof)
    
    unique_dofs = np.array(list(set(dofs)), dtype=np.int32)
    boundary_coords = V.tabulate_dof_coordinates()[unique_dofs] #surace node coordinates
    zmin=np.min(boundary_coords[:,1])
    print('-----zmin=',zmin)

    disp=Function(V) #displacement
    disp.assign(v_sol*dt)
    ALE.move(mesh, disp)

    ###################################################################################################
    ############################################ PLOT #################################################
    ###################################################################################################
    print('                                    >>> PLOTING results...')

    plotYlimMIN= H - 10
    plotYlimMAX = H

    if (tstep % nplot == 0 or (tstep*dt/yr2s>=1.96 and neemsaved==False) or tstep==(nsteps-1)):
            
        if (tstep*dt/yr2s>=1.96 and neemsaved==False):
            igifsold=igifs
            igifs='_1_96yrfinal'
            neemsaved=True
    
    
        #defining fonts
        plt.rcParams["font.family"] = "DejaVu Serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        plt.rcParams.update({'font.size': 18})
    

        #-----------------------    
        plt.figure()
        rhoplot = plot(rho_prev,cmap='PuBu',vmin=200,vmax=650)
        clb = plt.colorbar(rhoplot, orientation="vertical",label=r'Density (kgm$^{-3}$)',extend='both')
        plt.title(' k='+str(K)+'  '+str("%.2f"%(tstep*dt/yr2s))+'yr')
        plt.ylim(plotYlimMIN,plotYlimMAX+2)
        filename_rho = f'results_k{K}/{flagclimate}_{flagtrench}/zzzDensity{igifs}.png'
        plt.savefig(filename_rho, dpi=200, format='png')
        print(filename_rho)
        plt.close()

        #..................

        plt.figure()
        vplot = plot(v_sol*yr2s,cmap='jet')
        clb = plt.colorbar(vplot, orientation="vertical", label='V (m/yr)',extend='both')
        plt.title(' k='+str(K)+' '+' ('+str("%.2f"%(tstep*dt/yr2s))+'yr)')
        plt.ylim(plotYlimMIN,plotYlimMAX)
        filename_vel = f'results_k{K}/{flagclimate}_{flagtrench}/zzzVelocity{igifs}.png'
        plt.savefig(filename_vel, dpi=200, format='png')
        plt.close()

        #..................

        plt.figure()
        vplot = plot(v_sol*yr2s,cmap='jet')
        clb = plt.colorbar(vplot, orientation="vertical", label='V (m/yr)',extend='both')
        plt.title(' k='+str(K)+' '+' ('+str("%.2f"%(tstep*dt/yr2s))+'yr)')
        filename_vel = f'results_k{K}/{flagclimate}_{flagtrench}/zzzVelocity_ALL{igifs}.png'
        plt.savefig(filename_vel, dpi=200, format='png')
        plt.close()

        #.................. mesh

        plt.figure()
        plot(mesh, linewidth=0.25)
        plt.title(' k='+str(K)+' '+' ('+str("%.2f"%(tstep*dt/yr2s))+'yr)')
        plt.ylim(plotYlimMIN,plotYlimMAX)
        filename_mesh = f'results_k{K}/{flagclimate}_{flagtrench}/zzzMesh{igifs}.png'
        plt.savefig(filename_mesh, dpi=200, format='png')
        plt.close()

        #------------------------------temperature    

        plt.figure()
        rhoplot = plot(T_prev-273.15,cmap='jet',vmin=-60,vmax=0)
        plot(mesh,lw=0.25)
        clb = plt.colorbar(rhoplot, orientation="vertical", label='T_prev (ºC)',extend='both')
        plt.title(f'{K=} t={np.round(tstep*dt/yr2s,2)}yr  dt={np.round(dt/(60*60*24),2)}d---{tmr=}tunnel',fontsize=6)
        plt.ylim(plotYlimMIN,plotYlimMAX)
        plt.xlim(10,20)
        plt.savefig(f'results_k{K}/{flagclimate}_{flagtrench}/zzzTzoom{igifs}.png', dpi=200, format='png')
        plt.close()
        
        #-----------------------------------------
        plt.figure()
        rhoplot = plot(T_prev-273.15,cmap='jet',vmin=-60,vmax=0)
        plot(mesh,lw=0.25)
        clb = plt.colorbar(rhoplot, orientation="vertical", label='T_prev (ºC)',extend='both')
        plt.title(f'{K=} t={np.round(tstep*dt/yr2s,2)}yr  dt={np.round(dt/(60*60*24),2)}d---{tmr=}tunnel',fontsize=6)
        plt.savefig(f'results_k{K}/{flagclimate}_{flagtrench}/zzzT{igifs}.png', dpi=200, format='png')
        plt.close()

        #------
        plt.figure()
        plt.plot(times, hole_deltaxs, label='hole_deltaXs')
        plt.plot(times, hole_deltazs, label='hole_deltaZs')
        plt.ylabel('max dimension (m)')
        plt.xlabel('evolution time (yr)')
        plt.title(f'{K=} {dtfactor=}\n{tmsurf=} {hole_maxN=}\n{n_itacc=} {remeshstep=}',fontsize=7)
        plt.legend()
        plt.ylim(3.78,4.83)
        plt.xlim(0,2)
        plt.savefig(f'results_k{K}/{flagclimate}_{flagtrench}/_H_K{K}.png', dpi=200, bbox_inches='tight')
        plt.close()

        #---------------------------------

        plt.figure()
        plt.plot(steps_ts, avg_Tstep, c='tab:orange', label='Surface T')
        plt.plot(steps_ts[max_Ts!=0], max_Ts[max_Ts!=0], c='r',label='Maximum T')
        plt.plot(steps_ts[min_Ts!=0], min_Ts[min_Ts!=0], c='b',label='Minimum T')
        plt.title('temperature evolution')
        plt.legend(fontsize=5)
        plt.ylabel('T (ºC)')
        plt.savefig(f'results_k{K}/{flagclimate}_{flagtrench}/_Tminmax_evo_{flagclimate}_{flagtrench}.png',dpi=200, bbox_inches = 'tight')
        plt.close()


        # --------------------
        plt.figure()
        plt.plot(steps_ts, 550 * np.ones(nsteps), c='tab:orange', label='Rhotrench')
        plt.plot(steps_ts[max_RHOs != 0], max_RHOs[max_RHOs != 0], c='r', label='Maximum RHO')
        plt.plot(steps_ts[min_RHOs != 0], min_RHOs[min_RHOs != 0], c='b', label='Minimum RHO')
        plt.title(f'{K=} {dtfactor=}\n{tmsurf=} {hole_maxN=}\n{n_itacc=} {remeshstep=}', fontsize=7)
        plt.xlim(steps_ts[0],steps_ts[-1])
        plt.legend(fontsize=5)
        plt.ylabel('T (ºC)')
        plt.savefig(
            f'results_k{K}/{flagclimate}_{flagtrench}/_RHOminmax_evolution_K{K}.png',
            dpi=200, bbox_inches='tight')
        plt.close()

        # --------------------
        plt.figure()
        plt.plot(steps_ts[max_Vys!=0], max_Vys[max_Vys!=0], c='r',label='Maximum Vy')
        plt.plot(steps_ts[min_Vys!=0], min_Vys[min_Vys!=0], c='b',label='Minimum Vy')
        plt.plot(steps_ts[max_Vxs!=0], max_Vxs[max_Vxs!=0],ls='--', c='r',label='Maximum Vx')
        plt.plot(steps_ts[min_Vxs!=0], min_Vxs[min_Vxs!=0],ls='--', c='b',label='Minimum Vx')
        plt.title(f'{K=} {dtfactor=}\n{tmsurf=} {hole_maxN=}\n{n_itacc=} {remeshstep=}', fontsize=7)
        plt.xlim(steps_ts[0],steps_ts[-1])
        plt.legend(fontsize=5)
        plt.ylabel('T (ºC)')
        plt.savefig(f'results_k{K}/{flagclimate}_{flagtrench}/_Vminmax_evolution_K{K}.png',
            dpi=200, bbox_inches='tight')
        plt.close()

        #----------------------------------------
        #--update counter
        if isinstance(igifs, str):
            igifs=igifsold

        igifs += 1
        
        #------------------try clearing memory to avoid matplotlib related memory leaks
        #keep in mind and minimize the number of plots during the simulation if the process starts to get killed
        # without any clear error message
        plt.cla() # Clear the current axes.
        plt.clf() # Clear the current figure.
        plt.close('all') # Closes all the figure windows.
        #plt.close(fig)
        # gc.collect()

    #-------
    #if final iteration
    if (tstep*dt/yr2s>=1.96 and neemsaved==True):
        print('FINAL_____________SAVED_______BREAKING')
        break

    #update counter
    tstep += 1
    acc_iter=False


#____________________________________________________________________________________________out of the main loop
# save final results
np.save(f'results_k{K}/{flagclimate}_{flagtrench}/hole_times_{flagclimate}_{flagtrench}_k{K}.npy',times)
np.save(f'results_k{K}/{flagclimate}_{flagtrench}/hole_zmins_{flagclimate}_{flagtrench}_k{K}.npy',hole_zmins)
np.save(f'results_k{K}/{flagclimate}_{flagtrench}/hole_deltaxs_{flagclimate}_{flagtrench}_k{K}.npy',hole_deltaxs)
np.save(f'results_k{K}/{flagclimate}_{flagtrench}/hole_deltazs_{flagclimate}_{flagtrench}_k{K}.npy',hole_deltazs)

