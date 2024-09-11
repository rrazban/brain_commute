"""
Calculate metrics based on brain structure 
derived from diffusion MRI.


"""

import numpy as np


def laplacian(mat):
	degree = mat.sum(axis=0)
	mat = -mat
	np.fill_diagonal(mat, degree)
	return mat


def commute_time(structure):
	total_edges = structure.sum()

	lap = laplacian(structure)
	inv_lap = np.linalg.pinv(lap)

	size = lap.shape[0]
	dists = np.zeros((size, size))
	for i in range(size):
		for j in range(size):
			dists[i,j] = (inv_lap[i,i] + inv_lap[j,j] - 2*inv_lap[i,j])*total_edges		#technically, this is the defn of commute time
#			dists[i,j] = (inv_lap[i,i] + inv_lap[j,j] - 2*inv_lap[i,j])
#			dists[i,j] =  inv_lap[i,j]	#hack to calculate ENM cross correlation
	return dists

def hitting_time(adjacency):
    lap = laplacian(adjacency)
    inv_lap = np.linalg.pinv(lap)
    degree = adjacency.sum(axis=0)

    size = adjacency.shape[0]
    dists = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            for k in range(size):
#                dists[i,j] += (inv_lap[k,j]  - inv_lap[i,j]+  inv_lap[k,i]+inv_lap[i,i])*degree[k]  #not symmetric! interesting property
               dists[i,j] += (inv_lap[k,i]  - inv_lap[i,j]+  inv_lap[k,j]+inv_lap[j,j])*degree[k]
    return dists


def deconstruct_cov(mat, cutoff):
	evals,evecs=np.linalg.eig(mat)
	idx = evals.argsort()[::-1]


	ordered_evals = evals[idx]
	ordered_evecs = evecs[:,idx]

	size = mat.shape[0]
	new_mat = np.zeros((size, size))
	for i in range(size):
		for j in range(size):
			for k in range(cutoff):
				new_mat[i][j]+= ordered_evals[k] * ordered_evecs[i,k]*ordered_evecs[j,k]
	return new_mat 

def weighted_communicability(mat):
    size = mat.shape[0]
    degree = mat.sum(axis=0)

    comm = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            comm[i][j]=mat[i,j]/(degree[i]*degree[j])**(1/2)


    return np.exp(comm) 



def nx_path_length(adjacency):	
	size = adjacency.shape[0]
	dists = np.zeros((size, size))
	G = nx.from_numpy_array(adjacency)
	
	d_result = dict(nx.shortest_path_length(G))
	for key, result in d_result.items():
		for node, val in result.items():
			dists[key, node]= val	
	return dists



def search_information(W, L, has_memory=False):
    """
    SEARCH_INFORMATION: Search information

    SI = search_information(W, L, has_memory)

    Computes the amount of information (measured in bits) that a random
    walker needs to follow the shortest path between a given pair of nodes.

    Inputs:

        W
            Weighted/unweighted directed/undirected
            connection weight matrix.

        L
            Weighted/unweighted directed/undirected
            connection length matrix.

        has_memory
            This flag defines whether or not the random walker "remembers"
            its previous step, which has the effect of reducing the amount
            of information needed to find the next state. If this flag is
            not set, the walker has no memory by default.

    Outputs:

        SI
            pair-wise search information (matrix). Note that SI(i,j) may be
            different from SI(j,i), hence, SI is not a symmetric matrix
            even when adj is symmetric.

    References: Rosvall et al. (2005) Phys Rev Lett 94, 028701
                Goñi et al (2014) PNAS doi: 10.1073/pnas.131552911

    Andrea Avena-Koenigsberger and Joaquin Goñi, IU Bloomington, 2014
    Caio Seguin, University of Melbourne, 2019

    Modification history
    2014 - original
    2016 - included SPL transform option and generalized for
          symmetric/asymmetric networks
    2019 - modified to make user directly specify  weight-to-length transformations
    """

    if not has_memory:
        has_memory = False

    N = W.shape[0]

    if np.array_equal(W, W.T):
        flag_triu = True
    else:
        flag_triu = False

    T = np.linalg.solve(np.diag(np.sum(W, axis=1)), W)

    SPL, hops, Pmat = distance_wei_floyd(L)

    SI = np.zeros((N, N))
    np.fill_diagonal(SI, np.nan)

    for i in range(N):
        for j in range(N):
            if (j > i and flag_triu) or (not flag_triu and i != j):
                path = retrieve_shortest_path(i, j, hops, Pmat)
                lp = len(path)
                if flag_triu:
                    if path:
                        pr_step_ff = np.empty(lp-1)
                        pr_step_bk = np.empty(lp-1)
                        if has_memory:
                            pr_step_ff[0] = T[path[0], path[1]]
                            pr_step_bk[-1] = T[path[-1], path[-2]]
                            for z in range(1, lp-1):
                                pr_step_ff[z] = T[path[z], path[z+1]] / (1 - T[path[z-1], path[z]])
                                pr_step_bk[-z-1] = T[path[-z], path[-z-1]] / (1 - T[path[-z+1], path[-z]])
                        else:
                            for z in range(lp-1):
                                pr_step_ff[z] = T[path[z], path[z+1]]
                                pr_step_bk[z] = T[path[z+1], path[z]]
                        prob_sp_ff = np.prod(pr_step_ff)
                        prob_sp_bk = np.prod(pr_step_bk)
                        SI[i, j] = -np.log2(prob_sp_ff)
                        SI[j, i] = -np.log2(prob_sp_bk)
                    else:
                        SI[i, j] = np.inf
                        SI[j, i] = np.inf
                else:
                    if path:
                        pr_step_ff = np.empty(lp-1)
                        if has_memory:
                            pr_step_ff[0] = T[path[0], path[1]]
                            for z in range(1, lp-1):
                                pr_step_ff[z] = T[path[z], path[z+1]] / (1 - T[path[z-1], path[z]])
                        else:
                            for z in range(lp-1):
                                pr_step_ff[z] = T[path[z], path[z+1]]
                        prob_sp_ff = np.prod(pr_step_ff)
                        SI[i, j] = -np.log2(prob_sp_ff)
                    else:
                        SI[i, j] = np.inf

    return SI

def distance_wei_floyd(D, transform=None):
    """
    DISTANCE_WEI_FLOYD: Distance matrix (Floyd-Warshall algorithm)

    Computes the topological length of the shortest possible path
    connecting every pair of nodes in the network.

    Inputs:

        D,
            Weighted/unweighted directed/undirected 
            connection *weight* OR *length* matrix.

        transform,
            If the input matrix is a connection *weight* matrix, specify a
            transform that maps input connection weights to connection
            lengths. Two transforms are available:
                'log' -> l_ij = -log(w_ij)
                'inv' -> l_ij = 1 / w_ij

            If the input matrix is a connection *length* matrix, do not
            specify a transform (or specify an empty transform argument).

    Outputs:

        SPL,
            Unweighted/Weighted shortest path-length matrix.
            If W is a directed matrix, then SPL is not symmetric.

        hops,
            Number of edges in the shortest path matrix. If W is
            unweighted, SPL and hops are identical.				#confused! I am assuming that W refers to D

        Pmat,
            Elements {i,j} of this matrix indicate the next node in the
            shortest path between i and j. This matrix is used as an input
            argument for function 'retrieve_shortest_path', which returns
            as output the sequence of nodes comprising the shortest path
            between a given pair of nodes.

    Notes:

        There may be more than one shortest path between any pair of nodes
        in the network. Non-unique shortest paths are termed shortest path
        degeneracies and are most likely to occur in unweighted networks.
        When the shortest path is degenerate, the elements of matrix Pmat
        correspond to the first shortest path discovered by the algorithm.

        The input matrix may be either a connection weight matrix or a
        connection length matrix. The connection length matrix is typically
        obtained with a mapping from weight to length, such that higher
        weights are mapped to shorter lengths (see above).

    Algorithm: Floyd–Warshall Algorithm

    Andrea Avena-Koenigsberger, IU, 2012

    Modification history:
    2016 - included transform variable that maps weights to lengths
    """
    if transform:
        if transform == 'log':
            if np.any((D < 0) & (D > 1)):
                raise ValueError('Connection strengths must be in the interval [0, 1) to use the transform -log(w_ij)')
            else:
                SPL = -np.log(D)
        elif transform == 'inv':
            SPL = 1 / D
        else:
            raise ValueError('Unexpected transform type. Only "log" and "inv" are accepted')
    else:
        SPL = D.copy()
        SPL[SPL == 0] = np.inf

    n = D.shape[1]

    #if 'transform' in locals() and transform:
    if True:	#otherwise hops not defined!
        flag_find_paths = True
        hops = (D != 0).astype(np.double)
#        Pmat = np.arange(1, n + 1)[np.newaxis, :].repeat(n, axis=0)
        
        Pmat = np.arange(n)

# Replicate the row vector into a 2D matrix
        Pmat = np.tile(Pmat, (n, 1))

    else:
        flag_find_paths = False

    for k in range(n):
        i2k_k2j = SPL[:, k][:, np.newaxis] + SPL[k, :]
        
        if flag_find_paths:
            path = SPL > i2k_k2j
            i, j = np.where(path)
            hops[path] = hops[i, k] + hops[k, j].T
            Pmat[path] = Pmat[i, k]

        SPL = np.minimum(SPL, i2k_k2j)

    np.fill_diagonal(SPL, 0)

    if flag_find_paths:
        np.fill_diagonal(hops, 0)
        np.fill_diagonal(Pmat, 0)

    return SPL, hops, Pmat


def retrieve_shortest_path(s, t, hops, Pmat):
#meh, i could use a networkx module, i think. nvm, pmat takes into account distance 

    """
    RETRIEVE_SHORTEST_PATH: Retrieval of shortest path

    This function finds the sequence of nodes that comprise the shortest
    path between a given source and target node.

    Inputs:
        s
            Source node: i.e. node where the shortest path begins.
        t
            Target node: i.e. node where the shortest path ends.
        hops
            Number of edges in the path. This matrix may be obtained as the
            second output argument of the function "distance_wei_floyd".
        Pmat
            Pmat is a matrix whose elements {k,t} indicate the next node in
            the shortest path between k and t. This matrix may be obtained
            as the third output of the function "distance_wei_floyd".

    Output:
        path
            Nodes comprising the shortest path between nodes s and t.
    """
    path_length = hops[s, t]
    if path_length != 0:
        path = [None] * (int(path_length) + 1)
        path[0] = s
        for ind in range(1, len(path)):
            s = Pmat[s, t]
            path[ind] = s
    else:
        path = []
    return path



