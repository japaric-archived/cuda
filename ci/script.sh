set -ex

main() {
    cargo check --target $TARGET
}

main
