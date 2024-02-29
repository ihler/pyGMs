################################################################################
## IMPORTS #####################################################################
################################################################################


import numpy as np
from ipywidgets import Label, HTML, HBox, Image, VBox, Box, HBox
from ipyevents import Event
from IPython.display import display
import matplotlib.pyplot as plt
from PIL import Image as PIL_Image
import io
import networkx as nx 

from numpy import asarray as arr
from numpy import atleast_2d as twod


blank_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x03\x00\x08\xfc\x02\xfe\xa7\x9a\xa0\xa0\x00\x00\x00\x00IEND\xaeB`\x82'

def plot_to_widget(fig=None, data_only=False):
    """Convert pyplot figure to ipywidget Image object (or data for same), for interactive display"""
    #plt.tight_layout()
    if fig is None: fig=plt.gcf();
    fig.canvas.draw();
    with io.BytesIO() as output:
        fig.canvas.print_png(output)
        contents = output.getvalue()
    if data_only:
        return contents
    else:
        return Image(value=contents, format='png')



# Need to keep: nx Graph object (nodes, etc.); pos dict; labels (?)
#
# Allow to express: Name / label of node (?)
# Remove node (& all incident edges)
# Add node (type?)
# 


class draw_interactive(object):
    """Object to create & store graph from mouse input in (ipywidget+ipyevents) jupyter notebook
        Use mouse click to add nodes (hold combos of shift,alt,ctrl to set target value)
        Use drag to add edges
    """
    def __update_plot(self):
        fig=plt.figure(figsize=(6,6))
        plt.gcf().add_axes((self.border,self.border,1-2*self.border,1-2*self.border))  # full width figure
        plt.axis(self.ax); plt.gca().set_xticks([]); plt.gca().set_yticks([]);
        # TODO: draw graph here
        nx.draw(self._G, pos=self._pos[:self.M[0]], labels={i:i for i in range(self.M[0])},
          arrowstyle='->',arrowsize=10, node_color='w', edgecolors='k' )
        self.image.value = plot_to_widget(plt.gcf(), data_only=True);
        self.w,self.h = plt.gcf().canvas.get_renderer().get_canvas_width_height();
        #_=plt.clf();
        fig.clear();

    def __update_data(self,event):
        self.last_event[0] = event
        if (event['type']=='click'):          # left click = add point
            self._G.add_node(self.M[0])
            self._pos[self.M[0]] = self.event_position(event)
            self.M[0] += 1
        elif (event['type']=='contextmenu') and (self.m>0):  # right click = remove point
            if event['ctrlKey']: 
                self._src[0] = self.find_nearest(self.event_position(event))
            elif self._src[0] is not None:
                dst = self.find_nearest(self.event_position(event))
                if self._src[0] != dst: self._G.add_edge(self._src[0],dst)
                self._src[0] = None
        self.__update_plot()

    @property
    def G(self):
        return self._G
    @property
    def pos(self):
        return self._pos
    @property
    def m(self): return self.M[0]

    def event_position(self, event):
        return np.array([event['dataX']/self.w,1-event['dataY']/self.h])*2/(1-2*self.border) - 1-self.border
    
    def find_nearest(self,x):
        return ((self._pos[:self.m,:] - x.reshape(-1,2))**2).sum(1).argmin()
    
    def __init__(self, m=200, figsize=(6,6), plot=None):
        self.border = .01
        self.M = np.array([0]);
        self._G = nx.DiGraph()
        self._src = [None]
        #self._pos = { }
        self._pos = np.zeros((m,2))
        self.plot = plot;
        self.ax = [-1,1,-1,1];
        self.last_event = [None]
        self.image = Image(value=blank_png,format='png');
        self.__update_plot();

        self.no_drag = Event(source=self.image, watched_events=['dragstart'], prevent_default_action = True)
        self.events = Event(source=self.image, watched_events=['click','contextmenu'], prevent_default_action=True)
        self.events.on_dom_event(self.__update_data)
        if self.plot is None: print("Simple graph entry:\n  Left-click to add nodes; ctrl-rightclick sets source, rightclick adds edge (src,dst).\n  Use 'obj._repr_latex()' to generate tikz code.")

    def __repr__(self):
        return repr(self.image)
        
    def _ipython_display_(self):
        return self.image._ipython_display_()
        
    def _repr_latex(self):
        latex = "\\begin{tikzpicture} \n"
        latex+= "  \\tikzset{var/.style = {shape=circle,draw,minimum size=1.2em}} \n"
        latex+= "  \\tikzset{edge/.style = {->,> = latex,line width=0.5mm}} \n"
        rnd = np.round(self._pos,1)
        latex += "  \\foreach \\name/\\pos in {"+",".join(["{"+"{},({},{})".format(i,rnd[i,0],rnd[i,1])+"}" for i in range(self.m)])+"}\n"
        latex += "    \\node[var] (\\name) at \\pos {$\\name$}; \n"
        latex += "  \\foreach \\pa/\\ch in {"+",".join(["{"+"{},{}".format(e[0],e[1])+"}" for e in self._G.edges])+"}\n"
        latex += "    \\draw[edge] (\\pa) to (\\ch); \n"
        latex += "\\end{tikzpicture} \n"
        return latex
        
    def __str__(self):
        return "Mouse-based graph input; {} nodes ({} maximum)".format(self.m,len(self._pos)) #+"\n"+self.image.__str__()


################################################################################
class nxInteractiveLayout(object):
    """Object to create & store graph from mouse input in (ipywidget+ipyevents) jupyter notebook
        Use mouse click to add nodes (hold combos of shift,alt,ctrl to set target value)
        Use drag to add edges
    """
    def __update_plot(self):
        fig=plt.figure(figsize=(6,6))
        plt.gcf().add_axes((self.border,self.border,1-2*self.border,1-2*self.border))  # full width figure
        plt.axis(self.ax); plt.gca().set_xticks([]); plt.gca().set_yticks([]);
        # TODO: draw graph here
        nx.draw(self.G, pos=self.pos, labels={i:i for i in range(self.m)},
          arrowstyle='->',arrowsize=10, node_color='w', edgecolors='k' )
        if self._src[0] is not None:  # highlight selected node if any:
            nx.draw_networkx_nodes(self.G,pos=self.pos, nodelist=self._src, node_color='w', linewidths=2, edgecolors='r')
        self.image.value = plot_to_widget(plt.gcf(), data_only=True);
        self.w,self.h = plt.gcf().canvas.get_renderer().get_canvas_width_height();
        #_=plt.clf();
        fig.clear();

    def __update_data(self,event):
        if (event['type']=='click'):          # left click = add point
            if event['ctrlKey']:
              if self._src[0] is not None:
                self._pos[self._src[0]] = self.event_position(event)
            else:               
              self._G.add_node(self.m)
              self._pos[self.m] = self.event_position(event)
        elif (event['type']=='contextmenu') and (self.m>0):  # right click = select a node
            if event['ctrlKey']:
                if self._src[0] is not None:
                    dst = self.find_nearest(self.event_position(event))
                    if self._src[0] != dst: self._G.add_edge(self._src[0],dst)
                    self._src[0] = None
            else:    
                self._src[0] = self.find_nearest(self.event_position(event))
        self.__update_plot()

    @property
    def G(self):
        return self._G
    @property
    def pos(self):
        return self._pos
    @property
    def m(self): return len(self._pos)

    def event_position(self, event):
        return np.array([event['dataX']/self.w,1-event['dataY']/self.h])*2/(1-2*self.border) - 1-self.border
    
    def find_nearest(self,x):
        nodes = [key for key in self._pos]
        locat = np.array([self._pos[key] for key in self._pos]).reshape(-1,2)
        return nodes[((locat - x.reshape(-1,2))**2).sum(1).argmin()]
    
    def __init__(self, G, pos=None, figsize=(6,6)):
        self.border = .01
        self._G = G
        self._src = [None]
        self._pos = pos if pos is not None else { }
        self.ax = [-1,1,-1,1];
        self.image = Image(value=blank_png,format='png');
        self.__update_plot();

        self.no_drag = Event(source=self.image, watched_events=['dragstart'], prevent_default_action = True)
        self.events = Event(source=self.image, watched_events=['click','contextmenu'], prevent_default_action=True)
        self.events.on_dom_event(self.__update_data)
        print("Interactive graph layout:\n  Left-click to add nodes, Rightclick to select a node.\n  After selection, Control-Righclick adds an edge, while Control-Leftclick moves the selected node's location.\n Use 'obj._repr_latex()' to generate tikz code.")

    def __repr__(self):
        return repr(self._G)
        
    def _ipython_display_(self):
        return self.image._ipython_display_()
        
    def _repr_latex(self):
        latex = "\\begin{tikzpicture} \n"
        latex+= "  \\tikzset{var/.style = {shape=circle,draw,minimum size=1.2em}} \n"
        latex+= "  \\tikzset{edge/.style = {->,> = latex,line width=0.5mm}} \n"
        nodes = [key for key in self._pos]
        locat = np.array([self._pos[key] for key in self._pos]).reshape(-1,2).round(1)
        latex += "  \\foreach \\name/\\pos in {"+",".join(["{"+"{},({},{})".format(nodes[i],locat[i,0],locat[i,1])+"}" for i in range(len(locat))])+"}\n"
        latex += "    \\node[var] (\\name) at \\pos {$\\name$}; \n"
        latex += "  \\foreach \\pa/\\ch in {"+",".join(["{"+"{},{}".format(e[0],e[1])+"}" for e in self._G.edges])+"}\n"
        latex += "    \\draw[edge] (\\pa) to (\\ch); \n"
        latex += "\\end{tikzpicture} \n"
        return latex
        
    def __str__(self):
        return "Mouse-based graph input; {} nodes ({} maximum)".format(self.m,len(self._pos)) #+"\n"+self.image.__str__()



