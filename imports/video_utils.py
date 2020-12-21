from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import moviepy.editor as mp
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D
import matplotlib.cm as cm  



def max_weight(pac):
    # finds the highest value of the tau*eta matrix in across all iterations.
    # such a value is needed to set a alpha value (tramsparency) which never exceed 1
    return(np.max([np.max(pac[p][pac[p] != np.inf]) for p in pac.keys()]))



def find_line(idx_1,idx_2,a,b):
    # used to "track" the order in which lines have been initalised
    return(np.intersect1d(np.where(idx_1 == a)[0], np.where(idx_2 == b)[0]))
    
def best_tours_video(acs,results_folder,fps): 
    #during the simulation, anytime the current tour has improved the best so far, it is sotred
    #in this video, the sequence of such a tours are plotted at a rate of fps
    fig = plt.figure(figsize=(10,10))
    
    ic=acs.ic
    image_step=ic.nPoints
    pts=ic.points

    ax = fig.add_subplot(1, 1, 1,xlim=(np.min(ic.points[:,1])-1, np.max(ic.points[:,1])+1), 
                         ylim=(np.min(ic.points[:,2])-1, np.max(ic.points[:,2])+1))

    ax.set_title("Initial situation")
    tours=acs.global_best_tours.copy()

    for i in range(ic.nPoints): #plot the arcs with alpha = 0 (inititialization) 
        for j in range(i):
            ax.plot([pts[i, 1], pts[j, 1]],[pts[i, 2], pts[j, 2]], 'go-', linewidth=2,alpha=0)

    ind_p1,ind_p2=np.tril_indices(ic.nPoints,k=-1)  


    def animate_edges(k):
        for i,line in enumerate(ax.lines):
            line.set_alpha(0) # all edges made transparent
        T = tours[k]['optimal_tour'] # k-th best tour to be plotted
        T=np.append(T,T[0]) # to make the tour circular
        for i in range(len(T)-1):
            e = find_line(ind_p1,ind_p2,np.max([T[i],T[i+1]]),np.min([T[i],T[i+1]]))[0] # find the index of the edge between city T[i] and T[i+1]
            ax.lines[e].set_alpha(1)  # set the alha of such an edge equal to 1
        #set frame title
        ax.set_title("Found at iteration:" + str(tours[k]["iteration"]) + " - gap: " +str(np.round(tours[k]["gap"]*100,2)) + "%" )
       
        if k%100==0: #just to check the progress
            print("we are currently at step: ",k)
        return(line)


    #create the animation and save the video in the correct path
    anim = animation.FuncAnimation(fig, animate_edges,frames=len(tours))
    s=[results_folder,"video",'best_tour_evolution']
    fn=os.path.join(*s)
    
    anim.save(fn+'.gif', writer='imagemagick', fps=fps)
    clip = mp.VideoFileClip(fn+'.gif')
    clip.write_videofile(fn+'.mp4', fps=fps)
    
    
def pheromone_times_eta_video(acs,video_lenght, ant_no,results_folder):
    #video showing the temporal evolution of eta*tau

    fps = acs.iteration//video_lenght # fps to make teh video lasts as sp[ecified in the input 
    
    
    M=max_weight(acs.pher_eta_collection) # Max value of the temporal sequence tau*eta 
    
    fig = plt.figure(figsize=(10,10))

    ic=acs.ic
    image_step=ic.nPoints
    pts=ic.points

    ax = fig.add_subplot(1, 1, 1,xlim=(np.min(ic.points[:,1])-1, np.max(ic.points[:,1])+1), 
                         ylim=(np.min(ic.points[:,2])-1, np.max(ic.points[:,2])+1))

    ax.scatter(pts[:,1],pts[:,2], lw=2) #Plot cities
    
    ind_p1,ind_p2=np.tril_indices(ic.nPoints,k=-1)
    
    for i in range(ic.nPoints):
        for j in range(i):
            w=acs.pher_eta_collection['0_0'][ind_p1[i],ind_p2[i]]/M  #extract the normalised tau*edge value of the i-th edge 
            ax.plot([pts[i, 1], pts[j, 1]],[pts[i, 2], pts[j, 2]], 'g-', linewidth=2,alpha=w) #set the alpha value of the edge equal to the 

    def animate_edges(k):
        for i,line in enumerate(ax.lines):
            w=acs.pher_eta_collection[str(k)+'_0'][ind_p1[i],ind_p2[i]]/M  #extract the normalised tau*edge value of the i-th form the k-th matrix 
            line.set_alpha(w)
        if k%100==0:
            print("we are currently at step: ",k)
        return(line)

    #create the animation and save the video in the correct path
    s=[results_folder,"video",'pheromone_times_eta']
    fn=os.path.join(*s)
    
    anim = animation.FuncAnimation(fig, animate_edges,frames=acs.iteration)
    anim.save(fn+'.gif', writer='imagemagick', fps=fps)
    clip = mp.VideoFileClip(fn+".gif")
    clip.write_videofile(fn+".mp4", fps=fps)



def ant_tracking(acs, results_folder,iterations_to_show,fps=2):
    #video tracking step by step movements of an ant in the  iterations included in iterations_to_show
    fig = plt.figure(figsize=(10,10))
    
    ic=acs.ic
    image_step=ic.nPoints
    pts=ic.points
    
    point_colors={'non_visited':0.45, 
                'visited': 0.9,
                'current': 0.8}
    
    
    ax = fig.add_subplot(1, 1, 1,xlim=(np.min(ic.points[:,1])-1, np.max(ic.points[:,1])+1), 
                         ylim=(np.min(ic.points[:,2])-1, np.max(ic.points[:,2])*1.1))

    ax.title.set_text('Starting point')
    particles = ax.scatter(pts[:, 1], pts[:, 2], c=[point_colors['non_visited']]*ic.nPoints,s=100, cmap=cm.nipy_spectral, vmin=0, vmax=1,zorder=1) 
    next_city_pt = ax.scatter(pts[0, 1], pts[0, 2], s=300,facecolors='w', edgecolors='r',zorder=0) 

    for i in range(ic.nPoints):
        for j in range(i):
            ax.plot([pts[i, 1], pts[j, 1]],[pts[i, 2], pts[j, 2]], 'g-', linewidth=3,alpha=0,zorder=0) #plot edges (trarnsparent


    #create the legend
    handles = [Line2D([0], [0], marker='o', color='w', label='Scatter',
                          markerfacecolor='orange',markeredgecolor=cm.nipy_spectral(point_colors['current']), markersize=10),
               Line2D([0], [0], marker='o', color='w', label='Scatter',
                          markerfacecolor='r',markeredgecolor=cm.nipy_spectral(point_colors['visited']), markersize=10),
               Line2D([0], [0], marker='o', color='w', label='Scatter',
                          markerfacecolor='g',markeredgecolor=cm.nipy_spectral(point_colors['non_visited']), markersize=10),
               Line2D([0], [0], marker='o', color='w', label='Scatter',
                          markerfacecolor='none',markeredgecolor='r', markersize=10)]
    lgd=ax.legend(handles, ["Current","Visited","Available", "Next"],loc='top center', ncol=4)


    ind_p1,ind_p2=np.tril_indices(ic.nPoints,k=-1)

    #extract the info related only to the "iterations to show"
    steps=np.array(acs.local_step_info_collector[0]).flatten()
    sub_steps=[steps[it*ic.nPoints+j]  for it in iterations_to_show for j in range(ic.nPoints)]
    pac=[acs.pher_eta_collection[str(it)+'_'+str(j)] for it in iterations_to_show for j in range(ic.nPoints)]
        
    def animate_edges(k):
        for i,line in enumerate(ax.lines):
            line.set_alpha(0)
            
        M=np.max(pac[k][pac[k] != np.inf]) #find the max tau*eta value 
        p=np.array([0.0]*len(pts)) #array of colors
        current_city=sub_steps[k].current_city
        next_city=sub_steps[k].next_city
        #create frame title
        if sub_steps[k].end_tour!=True:
            title='Iteration: ' + str(sub_steps[k].iteration) +' - '+ sub_steps[k].title +' - '+'Current city: ' + str(sub_steps[k].current_city)  +' - '+'Next city: ' + str(sub_steps[k].next_city)  
        else:
            title='Iteration: ' + str(sub_steps[k].iteration) +' - '+ 'Current city: ' + str(sub_steps[k].current_city)  +' - '+ sub_steps[k].title 

        for j in range(ic.nPoints):
            if j == current_city:
                p[j] = point_colors['current']
            else:
                L_idx = np.min([current_city, j]) #lower index city
                H_idx = np.max([current_city, j]) #higher index city
                e = find_line(ind_p1,ind_p2,H_idx,L_idx)[0] # index of the line between H_idx and L_idx
                w = pac[k][L_idx,H_idx]/M  #extract the normalised edge value
                ax.lines[e].set_alpha(w) #set the edge transparecy accoring to w
                if j in sub_steps[k].tour:
                    p[j] = point_colors['visited'] # set the point color depending on if the city has been visited or not
                else:
                    p[j] = point_colors['non_visited']
        ax.title.set_text(title)
        next_city_pt.set_offsets(np.c_[pts[next_city,1],pts[next_city,2]]) #move the "next city"
        particles.set_array(p) #set the colors of the points

        if k%100==0:
            print("we are currently at step: ",k)
        return()

    
    #create the animation and save the video in the correct path
    anim = animation.FuncAnimation(fig, animate_edges,frames=len(sub_steps))

    s=[results_folder,"video",'ant_tour']
    fn=os.path.join(*s)
    
    anim.save(fn+'.gif', writer='imagemagick', fps=fps)
    clip = mp.VideoFileClip(fn+".gif")
    clip.write_videofile(fn+".mp4", fps=fps)

    



