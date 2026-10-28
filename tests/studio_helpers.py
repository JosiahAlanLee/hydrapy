"""Read Studio's published state instead of render-thread scratch state."""


def published_output(session):
    # Preview readbacks temporarily select other outputs on the render thread.
    # _publish replaces this snapshot only after those selections are restored.
    with session.lock:
        return session.state['output']
