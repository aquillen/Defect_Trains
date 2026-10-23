import numpy as np

import ngsolve
from ngsolve.webgui import Draw
import netgen.geom2d

import math
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# forms of CGLE
# partial_t A = A - (1 + i \beta) |A|^2 A + (1 + i \alpha) \Delta A        Aranson
# partial_t A = A - (b_3      - i)|A|^2 A + (1 + i b_1   ) \Delta A        Chate

def Aranson_to_Chate(alpha,beta):
    if (beta == 0.0):
        b_3 = np.inf
        b_1 = alpha
    else:
        b_3 = 1.0/np.abs(beta)  # can't be negative
        b_1 = -1*np.sign(beta)*alpha
    print(' b_1={:.2f} b_3 ={:.2f}'.format(b_1,b_3))
    return b_1,b_3

def Chate_to_Aranson(b_1,b_3):
    alpha = b_1
    if (b_3 == 0.0):
        beta = np.inf
    else:
        beta = -1.0/b_3
    if (b_3 <0):
        print('b_3<0!!!')  # should be positive
    print(' alpha={:.2f} beta ={:.2f}'.format(alpha,beta))
    return alpha,beta

# creating a complex element space from two real fields
# arguments:
#  mesh: ngsolve mesh
#  korder: order  
def mkfes(mesh,korder, boundary_list):
    Q = ngsolve.H1(mesh, order=korder,dirichlet=boundary_list) 
    #fes = ngsolve.H1(mesh, order=3, complex=True)  # We need a complex element!
    fes = Q*Q # two real fields
    #(ux,uy), (vx,vy) = fes.TnT() # trial and test functions!
    return fes
    #gfu = ngsolve.GridFunction(fes) # to hold the solution 

# create initial conditions with noise and seeds
# destroys whatever happens to be in gfu, only zero dirichlet possible here
# arguments:
#   gfu: grid function on finite element space for loading with initial conditions
#   nseeds: number of real 1 seeds,
#   sig:  a random dispersion for choosing random initial conditions
#   h: a width for gaussian peaks (the seeds)
#   A_r0, A_i0 offsets for real and complex parts
# returns:
#   nothing, but gfu.components are set
# globals:  none
def gen_init(gfu,mesh,nseeds,sig,h,A_r0 = 0.0, A_i0=0.0,holes=True):

    A_r = gfu.components[0]
    A_i = gfu.components[1]
    r1  = A_r.vec.CreateVector()  # create a vector, consistent with gfu
    r1.SetRandom(); # fill with random numbers
    r1 = sig*r1 # scale by sig
    r2  = A_i.vec.CreateVector()  # create a vector, consistent with gfu
    r2.SetRandom(); # fill with random numbers
    r2 = sig*r2 # scale by sig
    
    # add constants
    ccf = ngsolve.CF(A_r0)  # will be a coefficient function
    ccf_I = ngsolve.CF(A_i0)

    # use the bounding box to find min and max range of x and y values
    netgen_mesh = mesh.ngmesh
    p1,p2 = netgen_mesh.bounding_box
    #print(p1,p2)
    xxmin = min(p1[0],p2[0])
    xxmax = max(p1[0],p2[0])
    yymin = min(p1[1],p2[1])
    yymax = max(p1[1],p2[1])
    
    # seed with some +1 values
    for k in range(nseeds):
        x_p = np.random.uniform(xxmin,xxmax)  # random position
        y_p = np.random.uniform(yymin,yymax)
        rad2 = (ngsolve.x - x_p)*(ngsolve.x - x_p) + (ngsolve.y - y_p)*(ngsolve.y - y_p)
        if (holes == True):
            ccf = ccf*(1.0-ngsolve.exp(-0.5*rad2/h**2))  # add holes
        else:
            ccf += ngsolve.exp(-0.5*rad2/h**2)  #  a coefficient function
            # add peaks
        
    
    A_r.Set(ccf)   # set the seeds into gfu, real part
    A_i.Set(ccf_I)   # the imaginary part constant offset
    A_r.vec.data += r1  # add noise  to the real part
    A_i.vec.data += r2 # add noise  to the imaginary part
    #return nothing
    #gfu.components[1].vec.data += r1
    
       
# make the norm from real and complex parts of the grid function
def norm_gfu(gfu):
    A_r = gfu.components[0];
    A_i = gfu.components[1]
    norm_A = ngsolve.sqrt(A_r*A_r + A_i*A_i)  # using coefficient function
    return norm_A

#  make an angle from real and complex components of the grid function, we want something that goes between 0 and 2pi
def theta_gfu(gfu):
    A_r = gfu.components[0]
    A_i = gfu.components[1]
    theta = ngsolve.atan2(A_i,A_r)   # atan2 exists now, conventionally [-pi,pi]
    return ngsolve.IfPos(theta, theta, theta + 2*np.pi)
    
 
# make assembled operators on the finite element space
# arguments:
#  fes: finite element space (real,real) H1
#  alpha:  Diffusive parameter (of complex part) for complex Ginzburg-Landau equation
#    is equal to b_1 for Chate form and equal to alpha for Aranson form
#  dt: timestep
# returns:
#  invastar, a_lap, b_op: assembled operators needed for Crank Nicolson update
#  minv:  an assembled inverse mass matrix (used for non-linear operator split)
# everything here seems ok with Dirichlet BC
def mk_ops(fes,alpha,dt):

    (ux,uy), (vx,vy) = fes.TnT() # trial and test functions!

    #alpha must be defined here to make the Laplacian op
    #  alpha = c_1 = b_1 of the different forms
    #D_u  = (1 + 1.j*alpha)  # complex diffusion coefficient, Chate and Araonson forms

    # -Laplacian op
    a_lap = ngsolve.BilinearForm(fes, symmetric=True)  # holds Laplacian operator
    a_lap +=         ngsolve.grad(ux)*ngsolve.grad(vx)*ngsolve.dx   # real stays real
    a_lap +=         ngsolve.grad(uy)*ngsolve.grad(vy)*ngsolve.dx   # im stays im
    a_lap += -alpha* ngsolve.grad(uy)*ngsolve.grad(vx)*ngsolve.dx   # im goes into real
    a_lap +=  alpha* ngsolve.grad(ux)*ngsolve.grad(vy)*ngsolve.dx   # real to im
    a_lap.Assemble()  # needed for incremental Crank Nicolson update

    # mass operator
    m = ngsolve.BilinearForm(fes)  # holds mass matrix
    m += ux*vx*ngsolve.dx  + uy*vy*ngsolve.dx
    m.Assemble()

    minv = m.mat.Inverse(freedofs=fes.FreeDofs())  # inverse mass matrix!
    #print(f"m.mat.nze = {m.mat.nze}, a1.mat.nze={a.mat.nze}") # check that they are the same
    # these are the number of nonzero elements in the sparse matrices

    # B = M - 0.5*L*dt , needed for Crank Nicholson update L = Laplacian times diffusion coeff, M = mass matrix
    b_op = m.mat.CreateMatrix()
    b_op.AsVector().data     = m.mat.AsVector() - 0.5*dt * a_lap.mat.AsVector()
    #print(f"b.nze = {b.nze}")

    # A* = M + 0.5 * L * dt   needed for Crank Nicholson update
    astar = m.mat.CreateMatrix() # create a matrix in the form of m
    astar.AsVector().data = m.mat.AsVector() + 0.5*dt * a_lap.mat.AsVector()
    invastar = astar.Inverse(freedofs=fes.FreeDofs())
    #print(f"astar.nze={astar.nze}")
    
    return invastar, a_lap, b_op, minv


# create non-linear bilinear operator,
#    we could move linear terms into operator a if we wanted to
# create operator for the nonlinear function for CGLE
#    (Complex Ginzburg Landau equation)
# will be linearized on assembly
#    - this function can be differentiated so will linearize
# arguments:
#   fes: finite element system (real,real)
#   b_3:  is b_3 in Chate form or beta in Aranson/Winterbottom form of CGLE
#   uform:  if "Chate" then use Chate form, otherwise use Aranson form of CGLE
#   if using Chate form  (uform == 'Chate')
#     return (u-(b_3 -1j)*np.absolute(u)*np.absolute(u)*u)   # with u complex
#   if not Chate form then  (uform != 'Chate') then
#      return (u  - (1 + beta j) |u|^2 u)   b_3 -> beta
def mkgfun(fes,b_3,uform='Chate'):
    (ux,uy), (vx,vy) = fes.TnT() # trial and test functions
    gfun = ngsolve.BilinearForm(fes,nonassemble=True)
    # don't assemble so nonlinear part can be updated!
    gfun += ux*vx*ngsolve.dx + uy*vy*ngsolve.dx   # u part
    if (uform=='Chate'):
        gfun += (-b_3*ux - uy)*(ux*ux + uy*uy)*vx*ngsolve.dx
        # Re [-(b3 -1j )u^2 u] = -b3 u^2 u_re - u^2 u_im
        gfun += (-b_3*uy + ux)*(ux*ux + uy*uy)*vy*ngsolve.dx
        # Im [-(b3 -1j) u^2 u] = -b3 u^2 u_im + u^2 u_re
    else:   # Winterbottom or Aranson form  b_3 -> c_2 = \beta
        gfun += (-1*ux + b_3*uy)*(ux*ux + uy*uy)*vx*ngsolve.dx
        # Re [-(1 +i b_3 )u^2 u] = - u^2 u_re + b3 * u^2 u_im
        gfun += (-1*uy - b_3*ux)*(ux*ux + uy*uy)*vy*ngsolve.dx
        # Im [-(1 +i b_3) u^2 u] = - u^2 u_im - b3 * u^2 u_re
    return gfun
    
# a different way to do the non-linear operator
def mkgfun2(fes,b_3,uform='Chate'):
    sp_fun = ngsolve.GridFunction(fes)
    (ux,uy), (vx,vy) = fes.TnT() # trial and test functions
    gfun = ngsolve.BilinearForm(fes,nonassemble=True)
    gfun += ux*vx*ngsolve.dx + uy*vy*ngsolve.dx  # A part
    ux_old = sp_fun.components[0]
    uy_old = sp_fun.components[1]
    A2 = ux_old*ux_old + uy_old*uy_old # |A|^2 part
    if (uform=='Chate'):
        gfun += (-b_3*ux - uy)*A2*vx*ngsolve.dx
        gfun += (-b_3*uy + ux)*A2*vy*ngsolve.dx
    else:   # Winterbottom or Aranson form  b_3 -> c_2 = \beta
        gfun += (-1*ux + b_3*uy)*A2*vx*ngsolve.dx
        gfun += (-1*uy - b_3*ux)*A2*vy*ngsolve.dx
    return gfun,sp_fun

# the extra nonlinear function: u is the complex field, b_3 is the free parameter
#   (Chate form, -> b_3 = c_2 = beta if Winterbottom/Aranson form)
# this is used if carrying out the non-linear term by hand
def g(uu,b_3,uform='Chate'):
    if (uform=='Chate'):
        return (uu-(b_3 -  1j )*np.absolute(uu)*np.absolute(uu)*uu)
        # Chate's form for CGLE
    else:
        # beta = b_3
        return (uu-(1.0+b_3*1j)*np.absolute(uu)*np.absolute(uu)*uu)

# for making pngs
# get node points and triangles from a 2d ngsolve-netgen triangular mesh
# also return a refiner
# arguments:
#  mesh:  an NGsolve mesh
# returns:
#  xv,yv point positions
#  tri_tt: a python/matplotlib set of triangles
#  refiner: a refining function
def pts_tri(mesh):
    # find the x,y coordinates of the mesh
    xv = np.zeros(mesh.nv)
    yv = np.zeros(mesh.nv)
    k=0 # store nodes
    for v in mesh.vertices:
        #print(v,v.point)
        xv[k] = v.point[0]
        yv[k] = v.point[1]
        k+= 1
    # get a list of triangles in the mesh
    triangles = np.zeros((mesh.ne,3), dtype =int)  # number of elements
    k=0  # store triangles
    for el in mesh.Elements():
        triangles[k,:] = np.array([el.vertices[0].nr, el.vertices[1].nr, el.vertices[2].nr], dtype = int)
        k += 1
    tri_tt = tri.Triangulation(xv, yv,triangles)  # is a python triangle thing
    refiner = tri.UniformTriRefiner(tri_tt)  # refiner function
    return xv,yv,tri_tt,refiner,triangles

# helper routine for creating filenames
def mkpngname(root,index):
    fnum = ''
    if (index < 10):
        fnum += '0'
    if  (index < 100):
        fnum += '0'
    ofile = root +  fnum + '{:d}'.format(index)   + '.png'
    print(' ',ofile)
    return ofile
    
# if you want a png of mesh to look at
def draw_mesh(mesh,ofile='',figsize=(3,3)):
    fig,ax = plt.subplots(1,1,figsize=figsize,dpi=250)
    ax.set_aspect(1.0)
    xv,yv,tri_tt,refiner,triangles = pts_tri(mesh) # is fast, why not put it here
    triangles = np.zeros((mesh.ne,3), dtype =int)  # number of elements
    lw = 0.3
    linecolor = 'blue'
    k=0
    for el in mesh.Elements():
        triangles[k,:] = np.array([el.vertices[0].nr, el.vertices[1].nr, el.vertices[2].nr], dtype = int)
        k += 1
    for k in range(mesh.ne):
        x1 = xv[triangles[k,0]];y1 = yv[triangles[k,0]];
        x2 = xv[triangles[k,1]];y2 = yv[triangles[k,1]];
        x3 = xv[triangles[k,2]];y3 = yv[triangles[k,2]];
        ax.plot([x1,x2],[y1,y2],'-',color=linecolor,lw=lw)
        ax.plot([x1,x3],[y1,y3],'-',color=linecolor,lw=lw)
        ax.plot([x2,x3],[y2,y3],'-',color=linecolor,lw=lw)
        
    #ax.set_xlim([min(xv),max(xv)])
    #ax.set_ylim([min(yv),max(yv)])

    if (len(ofile)>3):
        plt.savefig(ofile)
    plt.show()

# we can use this to find max and min positions of piece of the boundary
# labelled with string bstring
# arguments:
#   mesh: ngsolve mesh
#   bstring:  boundary name string
# returns:
#   max and min in x and y (2d domain)
def bnd_range(mesh,bstring):
    xlist = np.zeros(0)
    ylist = np.zeros(0)
    for el in mesh.Elements(ngsolve.BND):
        if (el.mat == bstring):
            #print(el.mat) # this gives a string
            for v in el.vertices:
                p = mesh[v].point
                px = p[0]
                py = p[1]
                xlist = np.append(xlist,px)
                ylist = np.append(ylist,py)
            
    b_xmin = np.min(xlist)
    b_xmax = np.max(xlist)
    b_ymin = np.min(ylist)
    b_ymax = np.max(ylist)
    print(b_xmin,b_xmax,b_ymin,b_ymax)
    return b_xmin,b_xmax,b_ymin,b_ymax
    
    
# set boundary using a grid function gfu_bnd that contains something on all Dirichlet boundaries
# (defined on the finite element system) onto
# the grid function gfu without destroying any data that happens to be in the interior of gfu
# arguments:
#  gfu: grid function containing fields
#  gfu_bnd:  grid function containing boundary values
#  fes: finite element system
# returns nothing, however gfu is modified so that the all the boundary values
#  of gfu are set to those of gfu_bnd
#  on ALL the Dirichlet boundaries
def set_on_boundary(gfu,gfu_bnd,fes):
    free_dofs = fes.FreeDofs()  # free degrees of freedom (not boundary)
    for dof in range(fes.ndof):
        #print(free_dofs[dof])
        if (free_dofs[dof]== False):  # only set the dirichlet boundary dofs
            gfu.vec.data[dof] = gfu_bnd.vec[dof]

# for running the integration, incremental Crank Nicolson
# + op split for non-linear part
# arguments:
#  fes: finite element space
#  gfu: grid function field output on finite element space, containing initial conditions
#  scene: for drawing updates   output of Draw ngsolve.webgui
#  invastar, b_op or a_lap,minv :      # matrix operators on the finite element system (used dt, alpha)
#  nsamples:  number of outputs to store in the multidimensional data set
#  b_3: parameter for the complex ginzburg landau model (Chate form)
#    is equal to beta for Aranson form
#  dt:        timestep  which was also used in the matrix operators invastar and b
# optional to set parameters:
#  initial_cond:  initial gfu
#  t0: = 0  initial time
#  tdur:  duration of entire integration
#  nsamples:  number of integrations to store or update with within the integration duration
#  poutput: make pngs if True
#  gfun_op: if = True, we use non-linear op, otherwise do non-linear term by hand (not currently working as of last check 10/26)
#  uform:  form of CGLE, either 'Chate' or if you put something else then we would get the winterbottom form b_3 -> c_2 = beta
# returns
#  time at end
#  no longer: gfut: was  a multidimensional set of grid functions on finite element system that
#          holds time stepped results
#           at different times
#  globals:
#    g() if gfun_op = False
#    mkgfun() if gfun_op = True
def TimeStepping_CN(fes, gfu, scene, invastar, a_lap, minv, b_3, dt, \
            t0 = 0, tdur = 15,\
            nsamples = 200,poutput=False,ofileroot='',figparms=[],\
            gfun_op = True,uform='Chate'):
    
    # the initial condition is set already in gfu.vec.data
    cnt = 0; time = t0; tend = t0 + tdur
    sample_int = int(np.floor(tend / dt / nsamples)+1)  # nsamples is the number of outputs we want
    fcount = 0  # fileoutputs
    
    #gfut = ngsolve.GridFunction(gfu.space,multidim=0)  # creates gfut
    #gfut.AddMultiDimComponent(gfu.vec)  #  This makes it so we couild store a series of solutions
    
    gfun2 = True  # if you want to use mkgfun2 in instead of mkgfun

    if (gfun_op == True):
        if (gfun2 ==False):
            gfun = mkgfun(fes,b_3,uform = uform)  # make the non-linear ~bilinear form
        # this form works
        else:
            gfun,sp_fun = mkgfun2(fes,b_3,uform = uform)  # make the non-linear ~bilinear form
    else:  # we need some storage vectors
        res_re = gfu.components[0].vec.CreateVector()
        res_im = gfu.components[1].vec.CreateVector()
        A_r = gfu.components[0]
        A_i = gfu.components[1]
        gfu_temp = ngsolve.GridFunction(fes)

    res = gfu.vec.CreateVector() # is a base vector

    while (time < tend - 0.5 * dt):  # integrate until time is about equal to tend
    
        if (gfun_op == True): # with bi linear op
            if (gfun2 ==True):
                sp_fun.components[0].Set(gfu.components[0])
                sp_fun.components[1].Set(gfu.components[1])
                gfun.Assemble()
            #else: gfun.Assemble()  # assemble non-linear operator, apparently not necessary?

        # Crank Nicolson update
        res.data =  -dt*(a_lap.mat * gfu.vec)  # this is dt L u^n
        gfu.vec.data += invastar*res  # incremental version of Crank Nicolson
        # on diffusive term only, respects Dirichlet boundary
        
        # operator split add in non-linear part with op
        if (gfun_op == True): # with bi linear op
            #if (gfun2 ==True):
            #    sp_fun.components[0].Set(gfu.components[0])
            #    sp_fun.components[1].Set(gfu.components[1])
            #    gfun.Assemble()
            #else: gfun.Assemble()  # assemble non-linear operator, apparently not necessary?
            res_b = dt*(gfun.mat*gfu.vec)
            gfu.vec.data += minv*res_b    # incremental update,
            # First order explicit Eulerian (operator split)
            # minv was assembled with Dirichlet Boundary conditions,
            #that means this will not affect the Dirichlet boundary
            
        else: # by hand
            A_rpass = A_r.vec.FV().NumPy()[:]  # the real part of the field
            A_ipass = A_i.vec.FV().NumPy()[:]  # get the imaginary part
            gfunv  = g(A_rpass + 1j*A_ipass,b_3,uform=uform) # compute the nonlinear term, is a numpy complex array
            res_re.FV()[:] = dt*np.real(gfunv)  # add in non-linear function
            res_im.FV()[:] = dt*np.imag(gfunv)
            #A_r.vec.FV().NumPy()[:] += dt*np.real(gfunv) # add in non-linear function is bad because kills boundary
            #A_i.vec.FV().NumPy()[:] += dt*np.imag(gfunv)
            gfu_temp.components[0].vec.data = res_re
            gfu_temp.components[1].vec.data = res_im
            gfu.vec.data += ngsolve.Projector(fes.FreeDofs(), True) * gfu_temp.vec
            # add in non-linear part and respect Dirichlet boundary, works 8.14.25, fails  10.2.26, works as of 10/20/26!

        print("\r {:.3f}".format(time),end="")
        if (cnt % sample_int == 0):
            #gfut.AddMultiDimComponent(gfu.vec)  ! notice not storing any data!!!!!!
            scene.Redraw()
        cnt += 1; time += dt

        # kill integration if full of NaNs
        cval = gfu.components[0].vec.FV().NumPy()[0]
        if (math.isnan(cval) == True):
            print('NaN')
            time = tend
            break

        # make pngs if poutput==True
        if (cnt % sample_int == 0):  # only make figures sometimes
            if (poutput == True):
                root = ofileroot  # change this is you want a better file root name
                ofile = mkpngname(root,fcount)
                clabel = figparms[0]
                labelpos = figparms[1]
                figsize = figparms[2]
                cbar_shrink = figparms[3]
                disp(fes.mesh,gfu,time,ofile=ofile,draw_mesh=False,subdivn=2,\
                    label=clabel,labelpos=labelpos,figsize=figsize,cbar_shrink=cbar_shrink)
                fcount += 1
                
    return time


# we write our own routine to display amplitude and phase
# mesh need not be that associated with gfu as we use triangular interpolation
# if you are running this over and over again (like to make pngs)
# then to make this more efficient you need to find the mesh points and triangles beforehand
# the points have to be within the mesh
# calls: pts_tri
# arguments:
#   mesh:  ngsolve mesh
#   gfu:   Gridfunction
#   ofile:  if you want a png file this is the output file
#   draw_mesh: if True draw the mesh too
#   subdivn: controls subsampling for triangular mesh
#   figsize: if you need to adjust this
#   label: for labelling parameters
#   labelpos: for labelposition
#   cbarshrink: controls size of colorbars
def disp(mesh,gfu,t=0,ofile='',draw_mesh=False,subdivn=3,label='',labelpos=[0.1,0.1],\
        figsize=[7,2.5], cbar_shrink=0.4):
        # ,defects=False,epsi=0.01):  gave up on finding defects automatically
    xv,yv,tri_tt,refiner,triangles = pts_tri(mesh)
    #cbar_shrink = 0.4
    
    fig,axarr = plt.subplots(1,2,figsize=figsize,sharex=True,sharey=True,dpi=300)
    plt.subplots_adjust(left=0.09,right=0.95,bottom=0.01,top=0.99)
    
    # get the field values at all the vertices of the mesh
    # gfu does interpolation, gfu need not be on the same mesh as the one passed here
    # we could use a subdivided mesh to make these points
    A_rvals = np.zeros(mesh.nv)
    A_ivals = np.zeros(mesh.nv)
    A_r = gfu.components[0]
    A_i = gfu.components[1]

    # point positions and mesh used here
    for k in range(mesh.nv):
        A_rvals[k] = A_r(mesh(xv[k],yv[k]))  # get values of real and complex points at the nodes
        A_ivals[k] = A_i(mesh(xv[k],yv[k]))
        
    # refiner used here, subdiv set here
    A_norm =np.sqrt(A_rvals**2 + A_ivals**2)
    tri_refi, A_norm_refi = refiner.refine_field(A_norm, subdiv=subdivn)
        
    tri_refi, A_rvals_refi = refiner.refine_field(A_rvals, subdiv=subdivn)
    tri_refi, A_ivals_refi = refiner.refine_field(A_ivals, subdiv=subdivn)
    #A_norm_refi  = np.sqrt( A_rvals_refi**2 + A_ivals_refi**2)
    A_angle_refi = np.atan2(A_ivals_refi,A_rvals_refi)
    
    ax0 = axarr[0]; ax1 = axarr[1]
    ax0.set_aspect('equal'); ax1.set_aspect('equal')
    
    colors = ["blue", "green", "lawngreen", "lightseagreen","gold","coral","brown"]
    vmax = 1.2
    nodes = [0.0/vmax, 0.2/vmax, 0.4/vmax, 0.6/vmax, 0.8/vmax, 1.0/vmax,1.2/vmax]
        
    cmap2 = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

    ax0.tripcolor(tri_refi, A_norm_refi, cmap=cmap2,vmin=0,vmax=vmax)  # show field on triangles

    #clevs = np.linspace(0,1.2,7)
    #ax0.tricontour(tri_refi, A_norm_refi,levs=clevs,colors='black',linewidths=0.5 )
    # was turbo
    
    ax1.tripcolor(tri_refi, A_angle_refi, cmap='twilight_shifted',vmin=-np.pi,vmax=np.pi)  # show field on triangles
    
    #ccw = ax1.tricontour(tri_refi, A_angle_refi, levels = [0.0], colors='white', linewidths=0.5)  # add in phi=0 white line
    #ax1.tricontour(tri_refi,A_angle_refi,levels =[np.pi-0.05], colors='black', linewidths=1.5)
    #ax1.tricontour(tri_refi,A_angle_refi,levels =[-np.pi+0.05], colors='black', linewidths=1.5)
    
    field0 = ax0.get_children()[0]  # for making color bar in ax0
    field1 = ax1.get_children()[0]
    cb0= fig.colorbar(field0,shrink=cbar_shrink)  # colorbar!
    cb1= fig.colorbar(field1,shrink=cbar_shrink)
    #cb1.add_lines(ccw)
    cb1.set_ticks([-np.pi,0,np.pi], labels=[r'-$\pi$', '0', r'$\pi$'])
    title = ax0.set_title(f'ampl,  $t$ = {t:.2f}')  # time stamp
    title = ax1.set_title(f'angle')
    ax0.set_xlim([min(xv),max(xv)])
    ax0.set_ylim([min(yv),max(yv)])
    
    if (draw_mesh==True):  # show mesh
        linecolor='blue'
        lw = 0.3
        for k in range(mesh.ne):
            x1 = xv[triangles[k,0]];y1 = yv[triangles[k,0]];
            x2 = xv[triangles[k,1]];y2 = yv[triangles[k,1]];
            x3 = xv[triangles[k,2]];y3 = yv[triangles[k,2]];
            ax0.plot([x1,x2],[y1,y2],'-',color=linecolor,lw=lw)
            ax0.plot([x1,x3],[y1,y3],'-',color=linecolor,lw=lw)
            ax0.plot([x2,x3],[y2,y3],'-',color=linecolor,lw=lw)
            ax1.plot([x1,x2],[y1,y2],'-',color=linecolor,lw=lw)
            ax1.plot([x1,x3],[y1,y3],'-',color=linecolor,lw=lw)
            ax1.plot([x2,x3],[y2,y3],'-',color=linecolor,lw=lw)
            
    #if (defects == True):  # plot defects
    #    ii = np.logical_and(A_rvals*A_rvals < epsi, A_ivals*A_ivals < epsi)
    #    xd = xv[ii];yd = yv[ii];
    #    ax1.plot(xd,yd,'+', color='red')
    # giving up on finding local min because most algorithms are not going to work
    #  on a finite element system
    
    if (len(label)>3):
        xl = labelpos[0]; yl = labelpos[1]
        ax0.text(xl,yl,label,transform=ax0.transAxes,verticalalignment='top', horizontalalignment='left',fontsize=12)

    if (len(ofile)>3):
        plt.savefig(ofile)
        plt.close()
    else:
        plt.show()

