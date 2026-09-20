import numpy as np


def order_flow(book, reset_rows, representation="of"):
    """book: [N, side(bid/ask), level, field(price/qty)], float64.

    Bid: improved -> current qty, unchanged -> qty delta, worsened -> -old qty.
    Ask: improved (lower price) -> current qty, unchanged -> delta, worsened -> -old qty.
    Reset after gaps, segment changes, and split boundaries: no predecessor leakage.
    """
    p, q = book[..., 0], book[..., 1]
    delta = p[1:] - p[:-1]
    bid = np.where(delta[:, 0] > 0, q[1:, 0],
                   np.where(delta[:, 0] == 0, q[1:, 0]-q[:-1, 0], -q[:-1, 0]))
    ask = np.where(delta[:, 1] < 0, q[1:, 1],
                   np.where(delta[:, 1] == 0, q[1:, 1]-q[:-1, 1], -q[:-1, 1]))
    values = np.concatenate([bid, ask], axis=1) if representation == "of" else bid-ask
    values = np.concatenate([np.zeros((1, values.shape[1])), values])
    values[reset_rows] = 0
    return values
