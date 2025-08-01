/**
 * @file dataloader.h
 * @brief Functions to load MNIST-like dataset (images and labels)
 */

#ifndef DATALOADER_H
#define DATALOADER_H

#include <stdint.h>
#include <stdio.h>

/**
 * @struct Dataset
 * @brief Represents a dataset containing image-label pairs.
 */
typedef struct
{
    uint8_t *labels;    /**< Array of label values */
    uint8_t **images;   /**< 2D array of grayscale image data */
    size_t size; /**< Number of image-label pairs */
    size_t imageWidth;  /**< Pixels per row */
    size_t imageHeight; /**< Pixels per column */
} Dataset;

/**
 * @brief Loads the MNIST-like dataset from label and image files.
 *
 * This function reads binary files with expected MNIST format,
 * parses the label and image arrays, and returns a populated Dataset.
 *
 * @param labelPath Path to the labels file
 * @param imagesPath Path to the images file
 * @return A pointer to a dynamically allocated Dataset on success,
 *         or NULL on failure (use dataset_getError() to inspect).
 */
Dataset *loadDataset(const char *labelPath, const char *imagesPath);

/**
 * @brief Frees allocated dataset memory
 *
 * @param dataset Pointer to the dataset
 */
int closeDataset(Dataset *dataset);

/**
 * @brief Returns a human-readable error message from the last failure.
 *
 * This function provides insight into the cause of a failure
 *
 * @return A static string containing the most recent error message.
 */
const char *dataset_getError(void);

#endif // DATALOADER_H
