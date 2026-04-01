     print("\nConvergence table:")
        print(f"{'h':>10} {'nnodes':>10} {'errL2':>15} {'rateL2':>10} {'errH1s':>15} {'rateH1s':>10} {'errH1':>15} {'rateH1':>10}")
        for h, nn, eL2, rL2, eH1s, rH1s, eH1, rH1 in zip(
            mesh_sizes, nnodes, errsL2, ratesL2, errsH1s, ratesH1s, errsH1, ratesH1
        ):
            print(f"{h:10.4f} {nn:10d} {eL2:15.6e} {rL2:10.4f} {eH1s:15.6e} {rH1s:10.4f} {eH1:15.6e} {rH1:10.4f}")
