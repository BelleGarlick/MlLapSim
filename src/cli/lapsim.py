from lapsim import encoder


def parse_cli_args(args):
    if args.function == 'encode':
        if not args.src or not args.dest:
            raise Exception("Both --src and --dest fields must be used for this function")

        encoder.from_cli(
            args.src,
            args.dest,
            n_partitions=args.partitions,
            flip=args.flip,
            batch_size=args.batch_size,
            cores=args.cores,
            portion=args.portion,
            seed=args.seed
        )

    else:
        print(f"Invalid lapsim function: '{args.function}'. Choose from: 'splice', 'encode'")
