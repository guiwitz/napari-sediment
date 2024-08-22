from matplotlib.widgets import SpanSelector

class SelectRange:
    
    def __init__(self, parent, ax, single=False):
        
        self.ax = ax
        self.single = single
        self.canvas = ax.figure.canvas
        self.parent = parent
        self.myline1 = None
        self.myline2 = None
        self.min_pos = None
        self.max_pos = None
        
        self.span = SpanSelector(ax, onselect=self.onselect, direction='horizontal',
                                 interactive=True, props=dict(facecolor='blue', alpha=0.5))#, button=1)
        

    def onselect(self, min_pos, max_pos):
        
        if self.myline1 is not None:
            self.myline1.pop(0).remove()
        if self.myline2 is not None:
            self.myline2.pop(0).remove()
        
        min_max = [self.ax.lines[0].get_data()[1].min(),
                   self.ax.lines[0].get_data()[1].max()]

        self.myline2 = self.ax.plot([max_pos, max_pos], min_max, 'r')
        if not self.single:
            self.myline1 = self.ax.plot([min_pos, min_pos], min_max, 'r')
            self.min_pos = min_pos
        
        self.max_pos = max_pos
        
    def disconnect(self):
        self.span.disconnect_events()
        self.canvas.draw_idle()