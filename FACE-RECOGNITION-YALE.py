import cv2
import numpy as np
import os
import glob
path="yale/"
files=os.listdir(path)



loog=[]
loog_exp=[]
pca_model=[]
pca_model_exp=[]
expressions = ['happy','normal','sad','sleepy','surprised','wink']



def image_load(loog):
    for i in range(1,16):
        X=[]
        pic = f"yale/yale1/subject{i:02d}_*.jpg"
        pics = glob.glob(pic)
        count=0
        for p in pics:
            count+=1
            if count==11:
                break
            img=cv2.imread(p,cv2.IMREAD_GRAYSCALE)
            pic_v=img.flatten()
            X.append(pic_v)
        loog.append(X)


def pca(loog, pca_model):
    for i in range(0,15):
        temp = np.array(loog[i])
        mean = np.mean(temp, axis=0)
        mean_centered = temp - mean

        U, S, Vt = np.linalg.svd(mean_centered, full_matrices=False)
        eigen_val = (S**2) / (mean_centered.shape[0] - 1)
        eigen_vectors = Vt.T

        idx = np.argsort(eigen_val)[::-1]
        eigen_val = eigen_val[idx]
        eigen_vectors = eigen_vectors[:, idx]

        variance = eigen_val / np.sum(eigen_val)
        cumulative_variance = np.cumsum(variance)

        k = np.where(cumulative_variance >= 0.99)[0][0] + 1
        PCA_ = eigen_vectors[:, :k]

        pca_model.append({
            'mean': mean,
            'basis': PCA_
        })



def test_identify_person(loog,pca_model):
    p = f"yale/test1/*.jpg"
    pics = glob.glob(p)
    count=0
    for pic in pics:
         
    
        img=cv2.imread(pic,cv2.IMREAD_GRAYSCALE)
        test=img.flatten()
    
    
        min_err=float('inf')
        predicted=None
        for i in range(0,15):
            mean_testing=pca_model[i]['mean']
            basis_testing=pca_model[i]['basis']
            centered_mean_testing=test-mean_testing
            projection_testing=np.dot(centered_mean_testing,basis_testing)
            test_reconstructed_testing=np.dot(projection_testing,basis_testing.T)+mean_testing
            current_error=np.linalg.norm(test-test_reconstructed_testing)
        
            if current_error< min_err:
                min_err=current_error
                predicted=f"Person{i+1}"
        print(f"Image is:{pic}")
        
        print(f"Predicted is:{predicted}")
        print("  ")





def load_image_facial_expression(loog_exp,expressions):
    for i in range(0,6):
        X=[]    
        pic=f"yale/yale2/*_{expressions[i]}.jpg"
        pics=glob.glob(pic)
        count=0
        for p in pics:
            count+=1
            if count==11:
                break
            img=cv2.imread(p,cv2.IMREAD_GRAYSCALE)
            pic_v=img.flatten()
            X.append(pic_v)
        loog_exp.append(X)


def pca_facial_expression(loog_exp,pca_model_exp,expressions):
    for i in range(0,6):
        temp=np.array(loog_exp[i])
        mean=np.mean(temp,axis=0)
        
        mean_centered=mean-temp
        
        U,S,Vt=np.linalg.svd(mean_centered,full_matrices=False)
        eigen_val=(S**2)/(mean_centered.shape[0]-1)
        eigen_vectors=Vt.T

        idx=np.argsort(eigen_val)[::-1]
        eigen_val=eigen_val[idx]
        eigen_vectors=eigen_vectors[:,idx]

        variance=eigen_val/np.sum(eigen_val)
        cumulative_varience=np.cumsum(variance)

        k=np.where(cumulative_varience>=0.99)[0][0]+1
        PCA_=eigen_vectors[:,:k]

        pca_model_exp.append({
            'mean': mean,
            'basis': PCA_
        })


def testing_exp(loog_exp,pca_model_exp,expressions):
    p = f"yale/test2/*.jpg"
    pics = glob.glob(p)
    count=0
    for pic in pics:
         
    
        img=cv2.imread(pic,cv2.IMREAD_GRAYSCALE)
        test=img.flatten()
    
    
        min_err=float('inf')
        predicted=None
        for i in range(0,6):
            mean_testing=pca_model_exp[i]['mean']
            basis_testing=pca_model_exp[i]['basis']
            centered_mean_testing=test-mean_testing
            projection_testing=np.dot(centered_mean_testing,basis_testing)
            test_reconstructed_testing=np.dot(projection_testing,basis_testing.T)+mean_testing
            current_error=np.linalg.norm(test-test_reconstructed_testing)
        
            if current_error< min_err:
                min_err=current_error
                predicted=expressions[i]
        print(f"Image is:{pic}")
        
        print(f"Predicted is:{predicted}")
        print("  ")

print("TASK 1:")
print("")


image_load(loog)
pca(loog,pca_model)
test_identify_person(loog,pca_model)


print("TASK 2:")
print("  ")
load_image_facial_expression(loog_exp,expressions)
pca_facial_expression(loog_exp,pca_model_exp,expressions)
testing_exp(loog_exp,pca_model_exp,expressions)
