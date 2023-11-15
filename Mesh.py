# -*- coding: utf-8 -*-
"""
This module contains the mesh class. This class is the triangular surface where the fractal tree is grown. 
"""
import numpy as np
from scipy.spatial import cKDTree
import collections
#from tvtk.api import tvtk
#from tvtk.common import configure_input
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


class Mesh:
    """Class that contains the mesh where fractal tree is grown. It must be Wavefront .obj file. Be careful on how the normals are defined. It can change where an specified angle will go.
    
    
    Args:    
        filename (str): the path and filename of the .obj file with the mesh.

    Attributes:
        verts (array): a numpy array that contains all the nodes of the mesh. verts[i,j], where i is the node index and j=[0,1,2] is the coordinate (x,y,z).
        connectivity (array): a numpy array that contains all the connectivity of the triangles of the mesh. connectivity[i,j], where i is the triangle index and j=[0,1,2] is node index.
        normals (array): a numpy array that contains all the normals of the triangles of the mesh. normals[i,j], where i is the triangle index and j=[0,1,2] is normal coordinate (x,y,z).
        node_to_tri (dict): a dictionary that relates a node to the triangles that it is connected. It is the inverse relation of connectivity. The triangles are stored as a list for each node.
        tree (scipy.spatial.cKDTree): a k-d tree to compute the distance from any point to the closest node in the mesh.
        
    """
    def __init__(self,filename  = None, verts = None, connectivity = None):
        if filename is not None:
            verts, connectivity = self.loadOBJ(filename)
        self.verts=np.array(verts)
        self.connectivity=np.array(connectivity)
        self.normals=np.zeros(self.connectivity.shape)
        self.node_to_tri=collections.defaultdict(list)
        for i in range(len(self.connectivity)):
            for j in range(3):
                self.node_to_tri[self.connectivity[i,j]].append(i)
            u=self.verts[self.connectivity[i,1],:]-self.verts[self.connectivity[i,0],:]
            v=self.verts[self.connectivity[i,2],:]-self.verts[self.connectivity[i,0],:]
            n=np.cross(u,v)
            self.normals[i,:]=n/np.linalg.norm(n)
        self.tree=cKDTree(verts)
        self.centroids = (self.verts[self.connectivity[:,0],:] + self.verts[self.connectivity[:,1],:] + self.verts[self.connectivity[:,2],:])/3.

        
    def loadOBJ(self,filename):  
        """This function reads a .obj mesh file
        
        Args:
            filename (str): the path and filename of the .obj file.
            
        Returns:
             verts (array): a numpy array that contains all the nodes of the mesh. verts[i,j], where i is the node index and j=[0,1,2] is the coordinate (x,y,z).
             connectivity (array): a numpy array that contains all the connectivity of the triangles of the mesh. connectivity[i,j], where i is the triangle index and j=[0,1,2] is node index.
        """
        numVerts = 0  
        verts = []  
        norms = []   
        connectivity=[]
        for line in open(filename, "r"):  
            vals = line.split()
            if len(vals)>0:
                if vals[0] == "v":  
                    v = list(map(float, vals[1:4]))
                    verts.append(v)  
                if vals[0] == "vn":  
                    n = list(map(float, vals[1:4]))
                    norms.append(n)  
                if vals[0] == "f": 
                    con=[]
                    for f in vals[1:]:  
                        w = f.split("/")  
  #                      print w
                        # OBJ Files are 1-indexed so we must subtract 1 below  
                        con.append(int(w[0])-1)
                        numVerts += 1  
                    connectivity.append(con)
        return verts, connectivity

    def project_new_point(self, point, verts_to_search = 1):
        """This function receives a triangle and project it on the mesh in order to get the index of the triangle where
        the projected point lies

        Args:
            point (array): coordinates of the point to project.
            verts_to_search (int): the number of verts wich the point is going to be projected
        Returns:
             intriangle (int): the index of the triangle where the projected point lies. If the point is outside surface, intriangle=-1.
        """
        d, nodes = self.tree.query(point, verts_to_search)
        if verts_to_search > 1:
            for node in nodes:
                projected_point, intriangle, r, t = self.project_point_check(point, node)
                if intriangle != -1:
                    return projected_point, intriangle, r, t
        else:
            projected_point, intriangle, r, t = self.project_point_check(point, nodes)
        return projected_point, intriangle, r ,t


    def project_point_check(self, point, node):
        """This function projects any point to the surface defined by the mesh.
        
        Args:
            point (array): coordinates of the point to project.
            node (int): index of the most close node to the point
        Returns:
             projected_point (array): the coordinates of the projected point that lies in the surface.
             intriangle (int): the index of the triangle where the projected point lies. If the point is outside surface, intriangle=-1.
        """
        #Get the closest point
        # d, node=self.tree.query(point)
        #print d, node
        #Get triangles connected to that node
        triangles=self.node_to_tri[node]
        #print triangles
        #Compute the vertex normal as the avergage of the triangle normals.
        vertex_normal=np.sum(self.normals[triangles],axis=0)
        #Normalize
        vertex_normal=vertex_normal/np.linalg.norm(vertex_normal)
        #Project to the point to the closest vertex plane
        pre_projected_point=point-vertex_normal*np.dot(point-self.verts[node],vertex_normal)
        #Calculate the distance from point to plane (Closest point projection)
        CPP=[]
        for tri in triangles:
            CPP.append(np.dot(pre_projected_point-self.verts[self.connectivity[tri,0],:],self.normals[tri,:]))
        CPP=np.array(CPP)
     #   print 'CPP=',CPP
        triangles=np.array(triangles)
        #Sort from closest to furthest
        order=np.abs(CPP).argsort()
       # print CPP[order]
        #Check if point is in triangle
        intriangle=-1
        for o in order:
            i=triangles[o]
      #      print i
            projected_point=(pre_projected_point-CPP[o]*self.normals[i,:])
      #      print projected_point
            u=self.verts[self.connectivity[i,1],:]-self.verts[self.connectivity[i,0],:]
            v=self.verts[self.connectivity[i,2],:]-self.verts[self.connectivity[i,0],:]
            w=projected_point-self.verts[self.connectivity[i,0],:]
       #     print 'check ortogonality',np.dot(w,self.normals[i,:])
            vxw=np.cross(v,w)
            vxu=np.cross(v,u)
            uxw=np.cross(u,w)
            sign_r=np.dot(vxw,vxu)
            sign_t=np.dot(uxw,-vxu)
            r = t = -1
        #    print sign_r,sign_t            
            if sign_r>=0 and sign_t>=0:
                r=np.linalg.norm(vxw)/np.linalg.norm(vxu)
                t=np.linalg.norm(uxw)/np.linalg.norm(vxu)
             #   print 'sign ok', r , t
                if r<=1 and t<=1 and (r+t)<=1.001:
              #      print 'in triangle',i
                    intriangle = i
                    break
        return projected_point, intriangle, r, t
                
    def writeVTU(self,filename, verts, connectivity, scalars = None, vectors = None):
        tvtk.Triangle().cell_type
        tri_type = tvtk.Triangle().cell_type
        ug = tvtk.UnstructuredGrid(points=verts)
        ug.set_cells(tri_type, connectivity)
        
        if scalars is not None:
            ug.point_data.scalars = scalars
            ug.point_data.scalars.name = 'phi'
            
        if vectors is not None:
            ug.cell_data.vectors = vectors
            ug.cell_data.vectors.name = 'X'
    
        w = tvtk.XMLUnstructuredGridWriter(file_name=filename)
        configure_input(w,ug)
        w.write()
                
    def Bmatrix(self,element):
        nodeCoords = self.verts[self.connectivity[element]]
        e1 = (nodeCoords[1,:] - nodeCoords[0,:])/np.linalg.norm(nodeCoords[1,:] - nodeCoords[0,:])
        e2 = ((nodeCoords[2,:] - nodeCoords[0,:]) - np.dot((nodeCoords[2,:] - nodeCoords[0,:]),e1)*e1)
        e2 = e2/np.linalg.norm(e2) # normalize
        
        x21 = np.dot(nodeCoords[1,:] - nodeCoords[0,:],e1)
        x13 = np.dot(nodeCoords[0,:] - nodeCoords[2,:],e1)
        x32 = np.dot(nodeCoords[2,:] - nodeCoords[1,:],e1)
        
        y23 = np.dot(nodeCoords[1,:] - nodeCoords[2,:],e2)
        y31 = np.dot(nodeCoords[2,:] - nodeCoords[0,:],e2)
        y12 = np.dot(nodeCoords[0,:] - nodeCoords[1,:],e2)
        
        J = x13*y23 - y31*x32
        
        B = np.array([[y23, y31, y12],[x32, x13, x21]])
        
        return B, J        
        
    def gradient(self, element, u):
        nodeCoords = self.verts[self.connectivity[element]]
        e1 = (nodeCoords[1,:] - nodeCoords[0,:])/np.linalg.norm(nodeCoords[1,:] - nodeCoords[0,:])
        e2 = ((nodeCoords[2,:] - nodeCoords[0,:]) - np.dot((nodeCoords[2,:] - nodeCoords[0,:]),e1)*e1)
        e2 = e2/np.linalg.norm(e2) # normalize
        e3 = np.cross(e1,e2)
        
        x21 = np.dot(nodeCoords[1,:] - nodeCoords[0,:],e1)
        x13 = np.dot(nodeCoords[0,:] - nodeCoords[2,:],e1)
        x32 = np.dot(nodeCoords[2,:] - nodeCoords[1,:],e1)
        
        y23 = np.dot(nodeCoords[1,:] - nodeCoords[2,:],e2)
        y31 = np.dot(nodeCoords[2,:] - nodeCoords[0,:],e2)
        y12 = np.dot(nodeCoords[0,:] - nodeCoords[1,:],e2)
        
        
        B = np.array([[y23, y31, y12],[x32, x13, x21]])
        J = x13*y23 - y31*x32
        
        grad = np.zeros(3)
        grad[:2] = np.dot(B,u)/J
        
        R = np.vstack((e1,e2,e3)).T
       # Rinv = np.linalg.inv(R)
        
        
        return np.dot(R,grad)      
        
    def StiffnessMatrix(self,B,J):    
        return np.dot(B.T,B)/(2.*J)
    def MassMatrix(self,J):
       # return np.eye(3)*J/3.
        return np.array([[2.0,1.0,1.0],
                         [1.0,2.0,1.0],
                         [1.0,1.0,2.0]])*J/12
    def ForceVector(self,B,J,X):
        return np.dot(B.T,X)/2.
        
        
    def computeGeodesic(self, nodes, nodeVals, filename = None, K = None, M = None, dt = 10.0):
        nNodes = self.verts.shape[0]
        nElem = self.connectivity.shape[0]
        
#        K = sp.lil_matrix((nNodes, nNodes))
#        M = sp.lil_matrix((nNodes, nNodes))

        F = np.zeros((nNodes,1))
        
        u0 = np.zeros((nNodes,1))
        
        u0[nodes] = 1e6
        
        #dt = 10.0
        
        if (K is None) or (M is None):
            K = np.zeros((nNodes,nNodes))
            M = np.zeros((nNodes,nNodes))
            for el, tri in enumerate(self.connectivity):
                j, i = np.meshgrid(tri,tri)
                B, J = self.Bmatrix(el) 
                k = self.StiffnessMatrix(B,J)
                m = self.MassMatrix(J)
                K[i, j] += k
                M[i,j] += m
            
            
        
        activeNodes = list(range(nNodes))
        for known in nodes:
            activeNodes.remove(known)
            
        jActive, iActive = np.meshgrid(activeNodes, activeNodes)
        
        jKnown, iKnown = np.meshgrid(nodes, activeNodes)
        
        A1 = sp.csr.csr_matrix(M + dt*K)
        u = spsolve(A1, u0)[:,None]
      #  u = np.linalg.solve(M + dt*K,u0)
        
        Xs = np.zeros((nElem,3))
        Js = np.zeros((nElem,1))
        
        
        for k,tri in enumerate(self.connectivity):
            j, i = np.meshgrid(tri,tri)
            B, J = self.Bmatrix(k) 
            Js[k] = J
            X = self.gradient(k,u[tri,0])
            Xs[k,:] = X/np.linalg.norm(X)
            Xnr = np.dot(B,u[tri,0]) # not rotated
            Xnr /= np.linalg.norm(Xnr)
            f = self.ForceVector(B,J,Xnr)
            F[tri,0] -= f
        A2 = sp.csr.csr_matrix(K[iActive, jActive])
        AT = spsolve(A2, F[activeNodes,0]-np.dot(K[iKnown, jKnown],nodeVals))
      #  AT = np.linalg.solve(K[iActive, jActive],F[activeNodes,0]-np.dot(K[iKnown, jKnown],nodeVals))    
        
        ATglobal = np.zeros(nNodes)
        
        ATglobal[activeNodes] = AT
        ATglobal[nodes] = nodeVals

        if filename is not None:
            self.writeVTU(filename, self.verts, self.connectivity, ATglobal, Xs)
            
        return ATglobal, Xs
    
    def computeLaplace(self, nodes, nodeVals, filename = None):
        nNodes = self.verts.shape[0]
        nElem = self.connectivity.shape[0]
        
        K = np.zeros((nNodes,nNodes))
        F = np.zeros((nNodes,1))
        
            
        
        activeNodes = list(range(nNodes))
        for known in nodes:
            activeNodes.remove(known)
            
        jActive, iActive = np.meshgrid(activeNodes, activeNodes)
        
        jKnown, iKnown = np.meshgrid(nodes, activeNodes)
        
        
        
        Js = np.zeros((nElem,1))
        
        
        for k,tri in enumerate(self.connectivity):
            j, i = np.meshgrid(tri,tri)
            B, J = self.Bmatrix(k) 
            Js[k] = J
            k = self.StiffnessMatrix(B,J)
            K[i, j] += k
            
        T = np.linalg.solve(K[iActive, jActive],F[activeNodes,0]-np.dot(K[iKnown, jKnown],nodeVals))    
        
        Tglobal = np.zeros(nNodes)
        
        Tglobal[activeNodes] = T
        Tglobal[nodes] = nodeVals

        if filename is not None:
            self.writeVTU(filename, self.verts, self.connectivity, Tglobal, None)
            
        return Tglobal
    def computeLaplacian(self):
        nNodes = self.verts.shape[0]
        
        K = np.zeros((nNodes,nNodes))
        M = np.zeros((nNodes,nNodes))

        
        for k,tri in enumerate(self.connectivity):
            j, i = np.meshgrid(tri,tri)
            B, J = self.Bmatrix(k) 
            k = self.StiffnessMatrix(B,J)
            K[i, j] += k
            m = self.MassMatrix(J)
            M[i,j] += m
            

        return K,M




      
        
    
