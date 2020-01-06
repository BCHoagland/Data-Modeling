from visdom import Visdom

d = {}
viz = Visdom()

def get_points(x, y, name, showlegend=False):
        return dict(
            x=x,
            y=y,
            mode='markers',
            type='custom',
            name=name,
            showlegend=showlegend
        )

def get_line(x, y, name, color='#000', isFilled=False, fillcolor='transparent', width=2, showlegend=False):
        if isFilled:
            fill = 'tonexty'
        else:
            fill = 'none'

        return dict(
            x=x,
            y=y,
            mode='lines',
            type='custom',
            line=dict(
                color=color,
                width=width),
            fill=fill,
            fillcolor=fillcolor,
            name=name,
            showlegend=showlegend
        )

def scatter(points, name, clear=False):
    if (name not in d) or (clear == True):
        d[name] = {'x': [], 'y': []}
    
    # add x and y coords to dict
    x, y = zip(*points)
    d[name]['x'].extend(x)
    d[name]['y'].extend(y)

    # save all points to graph
    win = name + 'scatter'
    title = name
    data = [get_points(d[name]['x'], d[name]['y'], name)]
    layout = dict(
        title=title,
        xaxis={'title': 'Requests'},
        yaxis={'title': 'Instances'}
    )

    # plot graph
    viz._send({'data': data, 'layout': layout, 'win': win})

def line(x, y, name):
    if name not in d:
        d[name] = {'x': [], 'y': []}
    
    # add x and y coords to dict
    d[name]['x'].append(x)
    d[name]['y'].append(y)

    # save all points to graph
    win = name
    title = name
    data = [get_line(d[name]['x'], d[name]['y'], name)]
    layout = dict(
        title=title,
        xaxis={'title': 'Epoch'},
        yaxis={'title': 'Loss'}
    )

    # plot graph
    viz._send({'data': data, 'layout': layout, 'win': win})