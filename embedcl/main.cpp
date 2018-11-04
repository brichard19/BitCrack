#include <stdio.h>


bool create(char *inputFile, char *outputFile, char *symbol)
{
    char buf[1024];

    FILE *fpIn = fopen(inputFile, "rb");

    if(fpIn == NULL) {
        printf("Error opening '%s' for reading\n", inputFile);
        return false;
    }

    FILE *fpOut = fopen(outputFile, "w");

    if(fpOut == NULL) {
        printf("Error opening '%s' for writing\n", outputFile);
        fclose(fpIn);
        return false;
    }

    fprintf(fpOut, "char %s[] = {", symbol);

    size_t bytesRead = 0;
    while((bytesRead = fread(buf, 1, sizeof(buf), fpIn))) {
        for(int i = 0; i < bytesRead; i++) {
            fprintf(fpOut, "0x%x,", buf[i]);
        }
    }
    fprintf(fpOut, "0x00};\n");

    fclose(fpIn);
    fclose(fpOut);

    return true;
}

void usage()
{
    printf("Usage:\n");
    printf("<input_file> <output_file> <symbol>\n");
}

int main(int argc, char **argv)
{
    if(argc != 4) {
        usage();

        return 1;
    }

    if(create(argv[1], argv[2], argv[3])) {
        return 0;
    } else {
        return 1;
    }
}