from toolkit.tracks import splicer


def parse_cli_args(args):
    if args.function == 'splice':
        if not args.src or not args.dest:
            raise Exception("Both --src and --dest fields must be used for this function")

        splicer.from_cli(
            args.src,
            args.dest,
            spacing=args.spacing,
            batch_size=args.batch_size,
            cores=args.cores,
            portion=args.portion,
            seed=args.seed,
            precision=args.precision
        )

    else:
        print(f"Invalid tracks function: '{args.function}'. Choose from: 'splice'.")
