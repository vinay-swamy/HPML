#include <stdio.h>

int main(int argc, char *argv[]) {
    // argc is the number of command-line arguments
    // argv is an array of strings containing the arguments

    // Check if there are at least two arguments (including the program name)
    if (argc < 2) {
        printf("Usage: %s <argument1> [argument2] [argument3] ...\n", argv[0]);
        return 1; // Exit with an error code
    }

    // Access and use the command-line arguments
    printf("Program name: %s\n", argv[0]);

    for (int i = 1; i < argc; i++) {
        printf("Argument %d: %s\n", i, argv[i]);
    }

    return 0; // Exit successfully
}