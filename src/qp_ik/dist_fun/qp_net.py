from qp_ik.dist_fun.qp_base_net import QPBaseNet


def get_qp_net(method: str, **kwargs) -> QPBaseNet:
    if method == "rdf":
        from qp_ik.dist_fun.qp_rdf_net import QPRDFNet

        return QPRDFNet(**kwargs)
    elif method == "nsdf":
        from qp_ik.dist_fun.qp_nsdf_net import QPNSDFNet

        return QPNSDFNet(**kwargs)
    elif method == "neurg":
        raise NotImplementedError(
            "The CUDA-accelerated NeuRG method is not open-source yet and will be made open-source after the paper is accepted."
        )
    elif method == "jsdf":
        from qp_ik.dist_fun.qp_jsdf_net import QPJSDFNet

        return QPJSDFNet(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
