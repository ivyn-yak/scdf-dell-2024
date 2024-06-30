import React from "react";
import { useState } from "react";

const ImageUpload = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [imageBase64, setImageBase64] = useState("");
  const [category, setCategory] = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
      setImageBase64("");
      setCategory(null);
    } else {
      setSelectedImage(null);
      setPreview(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedImage) return;

    console.log(selectedImage.name);

    const formData = new FormData();
    formData.append("file", selectedImage);

    try {
      const response = await fetch("http://localhost:8000/", {
        method: "POST",
        body: formData,
      });
      const result = await response.json();
      if (result) {
        console.log("File uploaded successfully");
        setImageBase64(result.img);
        setCategory(result.category);
        console.log(result.category, result.ratio);
      } else {
        console.error("Upload failed:", result.error);
      }
    } catch (error) {
      console.error("Error uploading the file:", error);
    }
  };

  return (
    <div className="max-w-xl mx-auto mt-10">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="flex flex-col items-center">
          <label
            htmlFor="file-upload"
            className="cursor-pointer bg-blue-500 text-white py-2 px-4 rounded"
          >
            Choose Image
          </label>
          <input
            id="file-upload"
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleImageChange}
          />
        </div>

        <div className="flex gap-5 items-center justify-center">
          {preview && (
            <div className="flex">
              <img src={preview} alt="Preview" className="w-64 h-64 " />
            </div>
          )}

          {imageBase64 && (
            <div className="flex items-center">
              <div className="relative w-64 h-64">
                <img
                  src={preview}
                  alt="Base Image"
                  className="  absolute inset-0 w-full h-full"
                />
                <img
                  src={`data:image/png;base64,${imageBase64}`}
                  className="absolute inset-0 w-full h-full object-cover opacity-50"
                  alt="Mask"
                />
              </div>
            </div>
          )}
        </div>

        {category && (
          <div className="mt-2">
            <p>Category: {category}</p>
          </div>
        )}

        <button
          type="submit"
          className="w-full bg-green-500 text-white py-2 px-4 rounded"
        >
          Upload Image
        </button>
      </form>
    </div>
  );
};

export default ImageUpload;
