#include <stdio.h>

#include "dataloader.h"

const char grayscaleMap[] = ".:-=+*#%@";

int init();

int main(void)
{
    Dataset *dataset = loadDataset("../input/train-labels.idx1-ubyte",
                                   "../input/train-images.idx3-ubyte");
    if (dataset == NULL)
    {
        printf("Dataset error: %s", dataset_getError());
        return 1;
    }

    FILE *txt = fopen("../out.txt", "w");

    if (txt == NULL)
    {
        perror("u probably dumb but who knows");
        fclose(txt);
        return 1;
    }

    for (int i = 0; i < dataset->size; i++)
    {
        printf("printing image %d\n", i);
        fprintf(txt, "%u\n", dataset->labels[i]);
        fputc('\n', txt);
        for (int h = 0; h < dataset->imageHeight; h++)
        {
            for (int w = 0; w < dataset->imageHeight; w++)
            {
                int idx = (h * dataset->imageWidth) + w;
                int ascii = dataset->images[i][idx] / 29;
                fputc(grayscaleMap[ascii], txt);
            }
            fputc('\n', txt);
        }
        fputc('\n', txt);
    }

    fclose(txt);
    closeDataset(dataset);

    return 0;
}
